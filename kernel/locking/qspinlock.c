// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Queued spinlock
 *
 * (C) Copyright 2013-2015 Hewlett-Packard Development Company, L.P.
 * (C) Copyright 2013-2014,2018 Red Hat, Inc.
 * (C) Copyright 2015 Intel Corp.
 * (C) Copyright 2015 Hewlett-Packard Enterprise Development LP
 *
 * Authors: Waiman Long <longman@redhat.com>
 *          Peter Zijlstra <peterz@infradead.org>
 */

#include <linux/smp.h>
#include <linux/bug.h>
#include <linux/cpumask.h>
#include <linux/percpu.h>
#include <linux/hardirq.h>
#include <linux/memblock.h>
#include <linux/mutex.h>
#include <linux/prefetch.h>
#include <asm/byteorder.h>
#include <asm/qspinlock.h>
#include <trace/events/lock.h>

/*
 * Include queued spinlock statistics code
 */
#include "qspinlock_stat.h"

/*
 * The basic principle of a queue-based spinlock can best be understood
 * by studying a classic queue-based spinlock implementation called the
 * MCS lock. A copy of the original MCS lock paper ("Algorithms for Scalable
 * Synchronization on Shared-Memory Multiprocessors by Mellor-Crummey and
 * Scott") is available at
 *
 * https://bugzilla.kernel.org/show_bug.cgi?id=206115
 *
 * This queued spinlock implementation is based on the MCS lock, however to
 * make it fit the 4 bytes we assume spinlock_t to be, and preserve its
 * existing API, we must modify it somehow.
 *
 * In particular; where the traditional MCS lock consists of a tail pointer
 * (8 bytes) and needs the next pointer (another 8 bytes) of its own node to
 * unlock the next pending (next->locked), we compress both these: {tail,
 * next->locked} into a single u32 value.
 *
 * Since a spinlock disables recursion of its own context and there is a limit
 * to the contexts that can nest; namely: task, softirq, hardirq, nmi. As there
 * are at most 4 nesting levels, it can be encoded by a 2-bit number. Now
 * we can encode the tail by combining the 2-bit nesting level with the cpu
 * number. With one byte for the lock value and 3 bytes for the tail, only a
 * 32-bit word is now needed. Even though we only need 1 bit for the lock,
 * we extend it to a full byte to achieve better performance for architectures
 * that support atomic byte write.
 *
 * We also change the first spinner to spin on the lock bit instead of its
 * node; whereby avoiding the need to carry a node from lock to unlock, and
 * preserving existing lock API. This also makes the unlock code simpler and
 * faster.
 *
 * N.B. The current implementation only supports architectures that allow
 *      atomic operations on smaller 8-bit and 16-bit data types.
 *
 */

#define MAX_NODES	4

/*
 * On 64-bit architectures, the qnode structure will be 16 bytes in
 * size and four of them will fit nicely in one 64-byte cacheline. For
 * pvqspinlock, however, we need more space for extra data. To accommodate
 * that, we insert two more long words to pad it up to 32 bytes. IOW, only
 * two of them can fit in a cacheline in this case. That is OK as it is rare
 * to have more than 2 levels of slowpath nesting in actual use. We don't
 * want to penalize pvqspinlocks to optimize for a rare case in native
 * qspinlocks.
 */
struct qnode {
	struct qnode *next;
	int locked; /* 1 if lock acquired */
	int count;  /* nesting count */
#ifdef CONFIG_PARAVIRT_SPINLOCKS
	int			cpu;
	u8			state;
#endif
};

/*
 * The pending bit spinning loop count.
 * This heuristic is used to limit the number of lockword accesses
 * made by atomic_cond_read_relaxed when waiting for the lock to
 * transition out of the "== _Q_PENDING_VAL" state. We don't spin
 * indefinitely because there's no guarantee that we'll make forward
 * progress.
 */
#ifndef _Q_PENDING_LOOPS
#define _Q_PENDING_LOOPS	1
#endif

/*
 * Per-CPU queue node structures; we can never have more than 4 nested
 * contexts: task, softirq, hardirq, nmi.
 *
 * Exactly fits one 64-byte cacheline on a 64-bit architecture.
 *
 * PV doubles the storage and uses the second cacheline for PV state.
 */
static DEFINE_PER_CPU_ALIGNED(struct qnode, qnodes[MAX_NODES]);

/*
 * We must be able to distinguish between no-tail and the tail at 0:0,
 * therefore increment the cpu number by one.
 */

static inline __pure u32 encode_tail(int cpu, int idx)
{
	u32 tail;

	tail  = (cpu + 1) << _Q_TAIL_CPU_OFFSET;
	tail |= idx << _Q_TAIL_IDX_OFFSET; /* assume < 4 */

	return tail;
}

static inline __pure struct qnode *decode_tail(u32 tail)
{
	int cpu = (tail >> _Q_TAIL_CPU_OFFSET) - 1;
	int idx = (tail &  _Q_TAIL_IDX_MASK) >> _Q_TAIL_IDX_OFFSET;

	return per_cpu_ptr(&qnodes[idx], cpu);
}

static inline __pure
struct qnode *grab_qnode(struct qnode *base, int idx)
{
	return &base[idx];
}

#define _Q_LOCKED_PENDING_MASK (_Q_LOCKED_MASK | _Q_PENDING_MASK)

/**
 * set_pending - set the pending bit.
 * @lock: Pointer to queued spinlock structure
 *
 * *,0,* -> *,1,*
 *
 * For paravirt, the pending bit is used by the queue head vCPU to indicate
 * that it is actively spinning on the lock and no lock stealing is allowed.
 * Non-paravirt, the pending bit is used to avoid loading the extra node
 * cacheline in the likely contended case.
 */
static __always_inline void set_pending(struct qspinlock *lock)
{
#if _Q_PENDING_BITS == 8
	WRITE_ONCE(lock->pending, 1);
#else
	atomic_or(_Q_PENDING_VAL, &lock->val);
#endif
}

/**
 * clear_pending - clear the pending bit.
 * @lock: Pointer to queued spinlock structure
 *
 * *,1,* -> *,0,*
 */
static __always_inline void clear_pending(struct qspinlock *lock)
{
#if _Q_PENDING_BITS == 8
	WRITE_ONCE(lock->pending, 0);
#else
	atomic_andnot(_Q_PENDING_VAL, &lock->val);
#endif
}

/**
 * clear_pending_set_locked - take ownership and clear the pending bit.
 * @lock: Pointer to queued spinlock structure
 *
 * *,1,0 -> *,0,1
 *
 * Lock stealing is not allowed if this function is used.
 */
static __always_inline void clear_pending_set_locked(struct qspinlock *lock)
{
#if _Q_PENDING_BITS == 8
	WRITE_ONCE(lock->locked_pending, _Q_LOCKED_VAL);
#else
	atomic_add(-_Q_PENDING_VAL + _Q_LOCKED_VAL, &lock->val);
#endif
}

/**
 * trylock_clear_pending - try to take ownership and clear the pending bit
 * @lock: Pointer to queued spinlock structure
 *
 * 0,1,0 -> 0,0,1
 */
static __always_inline int trylock_clear_pending(struct qspinlock *lock)
{
#if _Q_PENDING_BITS == 8
	return !READ_ONCE(lock->locked) &&
	       (cmpxchg_acquire(&lock->locked_pending, _Q_PENDING_VAL,
				_Q_LOCKED_VAL) == _Q_PENDING_VAL);
#else
	int val = atomic_read(&lock->val);

	for (;;) {
		int old, new;

		if (val  & _Q_LOCKED_MASK)
			break;

		/*
		 * Try to clear pending bit & set locked bit
		 */
		old = val;
		new = (val & ~_Q_PENDING_MASK) | _Q_LOCKED_VAL;
		val = atomic_cmpxchg_acquire(&lock->val, old, new);

		if (val == old)
			return 1;
	}
	return 0;
#endif
}

/*
 * xchg_tail - Put in the new queue tail code word & retrieve previous one
 * @lock : Pointer to queued spinlock structure
 * @tail : The new queue tail code word
 * Return: The previous queue tail code word
 *
 * xchg(lock, tail), which heads an address dependency
 *
 * p,*,* -> n,*,* ; prev = xchg(lock, node)
 */
static __always_inline u32 xchg_tail(struct qspinlock *lock, u32 tail)
{
	/*
	 * We can use relaxed semantics since the caller ensures that the
	 * MCS node is properly initialized before updating the tail.
	 */
#if _Q_PENDING_BITS == 8
	return (u32)xchg_relaxed(&lock->tail,
				 tail >> _Q_TAIL_OFFSET) << _Q_TAIL_OFFSET;
#else
	u32 old, new, val = atomic_read(&lock->val);

	for (;;) {
		new = (val & _Q_LOCKED_PENDING_MASK) | tail;
		old = atomic_cmpxchg_relaxed(&lock->val, val, new);
		if (old == val)
			break;

		val = old;
	}
	return old;
#endif
}

/**
 * queued_fetch_set_pending_acquire - fetch the whole lock value and set pending
 * @lock : Pointer to queued spinlock structure
 * Return: The previous lock value
 *
 * *,*,* -> *,1,*
 */
#ifndef queued_fetch_set_pending_acquire
static __always_inline u32 queued_fetch_set_pending_acquire(struct qspinlock *lock)
{
	return atomic_fetch_or_acquire(_Q_PENDING_VAL, &lock->val);
}
#endif

/**
 * set_locked - Set the lock bit and own the lock
 * @lock: Pointer to queued spinlock structure
 *
 * *,*,0 -> *,0,1
 */
static __always_inline void set_locked(struct qspinlock *lock)
{
	WRITE_ONCE(lock->locked, _Q_LOCKED_VAL);
}

#ifdef CONFIG_PARAVIRT_SPINLOCKS
/*
 * Implement paravirt qspinlocks; the general idea is to halt the vcpus instead
 * of spinning them.
 *
 * This relies on the architecture to provide two paravirt hypercalls:
 *
 *   pv_wait(u8 *ptr, u8 val) -- suspends the vcpu if *ptr == val
 *   pv_kick(cpu)             -- wakes a suspended vcpu
 *
 * Using these we implement __pv_queued_spin_lock_slowpath() and
 * __pv_queued_spin_unlock().
 */

#define _Q_SLOW_VAL	(3U << _Q_LOCKED_OFFSET)

/*
 * Queue Node Adaptive Spinning
 *
 * A queue node vCPU will stop spinning if the vCPU in the previous node is
 * not running. The one lock stealing attempt allowed at slowpath entry
 * mitigates the slight slowdown for non-overcommitted guest with this
 * aggressive wait-early mechanism.
 *
 * The status of the previous node will be checked at fixed interval
 * controlled by PV_PREV_CHECK_MASK. This is to ensure that we won't
 * pound on the cacheline of the previous node too heavily.
 */
#define PV_PREV_CHECK_MASK	0xff

/*
 * Queue node uses: vcpu_running & vcpu_halted.
 * Queue head uses: vcpu_running & vcpu_hashed.
 */
enum vcpu_state {
	vcpu_running = 0,
	vcpu_halted,		/* Used only in pv_wait_node */
	vcpu_hashed,		/* = pv_hash'ed + vcpu_halted */
};

/*
 * Hybrid PV queued/unfair lock
 *
 * This function is called once when a lock waiter enters the PV slowpath
 * before being queued.
 *
 * The pending bit is set by the queue head vCPU of the MCS wait queue in
 * pv_wait_head_or_lock() to signal that it is ready to spin on the lock.
 * When that bit becomes visible to the incoming waiters, no lock stealing
 * is allowed. The function will return immediately to make the waiters
 * enter the MCS wait queue. So lock starvation shouldn't happen as long
 * as the queued mode vCPUs are actively running to set the pending bit
 * and hence disabling lock stealing.
 *
 * When the pending bit isn't set, the lock waiters will stay in the unfair
 * mode spinning on the lock unless the MCS wait queue is empty. In this
 * case, the lock waiters will enter the queued mode slowpath trying to
 * become the queue head and set the pending bit.
 *
 * This hybrid PV queued/unfair lock combines the best attributes of a
 * queued lock (no lock starvation) and an unfair lock (good performance
 * on not heavily contended locks).
 */
static inline bool pv_hybrid_queued_unfair_trylock(struct qspinlock *lock)
{
	/*
	 * Stay in unfair lock mode as long as queued mode waiters are
	 * present in the MCS wait queue but the pending bit isn't set.
	 */
	for (;;) {
		int val = atomic_read(&lock->val);

		if (!(val & _Q_LOCKED_PENDING_MASK) &&
		   (cmpxchg_acquire(&lock->locked, 0, _Q_LOCKED_VAL) == 0)) {
			lockevent_inc(pv_lock_stealing);
			return true;
		}
		if (!(val & _Q_TAIL_MASK) || (val & _Q_PENDING_MASK))
			break;

		cpu_relax();
	}

	return false;
}

/*
 * Lock and MCS node addresses hash table for fast lookup
 *
 * Hashing is done on a per-cacheline basis to minimize the need to access
 * more than one cacheline.
 *
 * Dynamically allocate a hash table big enough to hold at least 4X the
 * number of possible cpus in the system. Allocation is done on page
 * granularity. So the minimum number of hash buckets should be at least
 * 256 (64-bit) or 512 (32-bit) to fully utilize a 4k page.
 *
 * Since we should not be holding locks from NMI context (very rare indeed) the
 * max load factor is 0.75, which is around the point where open addressing
 * breaks down.
 *
 */
struct pv_hash_entry {
	struct qspinlock *lock;
	struct qnode   *node;
};

#define PV_HE_PER_LINE	(SMP_CACHE_BYTES / sizeof(struct pv_hash_entry))
#define PV_HE_MIN	(PAGE_SIZE / sizeof(struct pv_hash_entry))

static struct pv_hash_entry *pv_lock_hash;
static unsigned int pv_lock_hash_bits __read_mostly;

/*
 * Allocate memory for the PV qspinlock hash buckets
 *
 * This function should be called from the paravirt spinlock initialization
 * routine.
 */
void __init __pv_init_lock_hash(void)
{
	int pv_hash_size = ALIGN(4 * num_possible_cpus(), PV_HE_PER_LINE);

	if (pv_hash_size < PV_HE_MIN)
		pv_hash_size = PV_HE_MIN;

	/*
	 * Allocate space from bootmem which should be page-size aligned
	 * and hence cacheline aligned.
	 */
	pv_lock_hash = alloc_large_system_hash("PV qspinlock",
					       sizeof(struct pv_hash_entry),
					       pv_hash_size, 0,
					       HASH_EARLY | HASH_ZERO,
					       &pv_lock_hash_bits, NULL,
					       pv_hash_size, pv_hash_size);
}

#define for_each_hash_entry(he, offset, hash)						\
	for (hash &= ~(PV_HE_PER_LINE - 1), he = &pv_lock_hash[hash], offset = 0;	\
	     offset < (1 << pv_lock_hash_bits);						\
	     offset++, he = &pv_lock_hash[(hash + offset) & ((1 << pv_lock_hash_bits) - 1)])

static struct qspinlock **pv_hash(struct qspinlock *lock, struct qnode *node)
{
	unsigned long offset, hash = hash_ptr(lock, pv_lock_hash_bits);
	struct pv_hash_entry *he;
	int hopcnt = 0;

	for_each_hash_entry(he, offset, hash) {
		hopcnt++;
		if (!cmpxchg(&he->lock, NULL, lock)) {
			WRITE_ONCE(he->node, node);
			lockevent_pv_hop(hopcnt);
			return &he->lock;
		}
	}
	/*
	 * Hard assume there is a free entry for us.
	 *
	 * This is guaranteed by ensuring every blocked lock only ever consumes
	 * a single entry, and since we only have 4 nesting levels per CPU
	 * and allocated 4*nr_possible_cpus(), this must be so.
	 *
	 * The single entry is guaranteed by having the lock owner unhash
	 * before it releases.
	 */
	BUG();
}

static struct qnode *pv_unhash(struct qspinlock *lock)
{
	unsigned long offset, hash = hash_ptr(lock, pv_lock_hash_bits);
	struct pv_hash_entry *he;
	struct qnode *node;

	for_each_hash_entry(he, offset, hash) {
		if (READ_ONCE(he->lock) == lock) {
			node = READ_ONCE(he->node);
			WRITE_ONCE(he->lock, NULL);
			return node;
		}
	}
	/*
	 * Hard assume we'll find an entry.
	 *
	 * This guarantees a limited lookup time and is itself guaranteed by
	 * having the lock owner do the unhash -- IFF the unlock sees the
	 * SLOW flag, there MUST be a hash entry.
	 */
	BUG();
}

/*
 * Return true if when it is time to check the previous node which is not
 * in a running state.
 */
static inline bool
pv_wait_early(struct qnode *prev, int loop)
{
	if ((loop & PV_PREV_CHECK_MASK) != 0)
		return false;

	return READ_ONCE(prev->state) != vcpu_running;
}

/*
 * Initialize the PV part of the qnode.
 */
static void pv_init_node(struct qnode *node)
{
	node->cpu = smp_processor_id();
	node->state = vcpu_running;
}

/*
 * Wait for node->locked to become true, halt the vcpu after a short spin.
 * pv_kick_node() is used to set _Q_SLOW_VAL and fill in hash table on its
 * behalf.
 */
static void pv_wait_node(struct qnode *node, struct qnode *prev)
{
	int loop;
	bool wait_early;

	for (;;) {
		for (wait_early = false, loop = SPIN_THRESHOLD; loop; loop--) {
			if (READ_ONCE(node->locked))
				return;
			if (pv_wait_early(prev, loop)) {
				wait_early = true;
				break;
			}
			cpu_relax();
		}

		/*
		 * Order node->state vs node->locked thusly:
		 *
		 * [S] node->state = vcpu_halted  [S] next->locked = 1
		 *     MB                             MB
		 * [L] node->locked             [RmW] node->state = vcpu_hashed
		 *
		 * Matches the cmpxchg() from pv_kick_node().
		 */
		smp_store_mb(node->state, vcpu_halted);

		if (!READ_ONCE(node->locked)) {
			lockevent_inc(pv_wait_node);
			lockevent_cond_inc(pv_wait_early, wait_early);
			pv_wait(&node->state, vcpu_halted);
		}

		/*
		 * If pv_kick_node() changed us to vcpu_hashed, retain that
		 * value so that pv_wait_head_or_lock() knows to not also try
		 * to hash this lock.
		 */
		cmpxchg(&node->state, vcpu_halted, vcpu_running);

		/*
		 * If the locked flag is still not set after wakeup, it is a
		 * spurious wakeup and the vCPU should wait again. However,
		 * there is a pretty high overhead for CPU halting and kicking.
		 * So it is better to spin for a while in the hope that the
		 * MCS lock will be released soon.
		 */
		lockevent_cond_inc(pv_spurious_wakeup,
				  !READ_ONCE(node->locked));
	}

	/*
	 * By now our node->locked should be 1 and our caller will not actually
	 * spin-wait for it. We do however rely on our caller to do a
	 * load-acquire for us.
	 */
}

/*
 * Called after setting next->locked = 1 when we're the lock owner.
 *
 * Instead of waking the waiters stuck in pv_wait_node() advance their state
 * such that they're waiting in pv_wait_head_or_lock(), this avoids a
 * wake/sleep cycle.
 */
static void pv_kick_node(struct qspinlock *lock, struct qnode *node)
{
	/*
	 * If the vCPU is indeed halted, advance its state to match that of
	 * pv_wait_node(). If OTOH this fails, the vCPU was running and will
	 * observe its next->locked value and advance itself.
	 *
	 * Matches with smp_store_mb() and cmpxchg() in pv_wait_node()
	 *
	 * The write to next->locked in arch_mcs_spin_unlock_contended()
	 * must be ordered before the read of node->state in the cmpxchg()
	 * below for the code to work correctly. To guarantee full ordering
	 * irrespective of the success or failure of the cmpxchg(),
	 * a relaxed version with explicit barrier is used. The control
	 * dependency will order the reading of node->state before any
	 * subsequent writes.
	 */
	smp_mb__before_atomic();
	if (cmpxchg_relaxed(&node->state, vcpu_halted, vcpu_hashed)
	    != vcpu_halted)
		return;

	/*
	 * Put the lock into the hash table and set the _Q_SLOW_VAL.
	 *
	 * As this is the same vCPU that will check the _Q_SLOW_VAL value and
	 * the hash table later on at unlock time, no atomic instruction is
	 * needed.
	 */
	WRITE_ONCE(lock->locked, _Q_SLOW_VAL);
	(void)pv_hash(lock, node);
}

/*
 * Wait for l->locked to become clear and acquire the lock;
 * halt the vcpu after a short spin.
 * __pv_queued_spin_unlock() will wake us.
 *
 * The current value of the lock will be returned for additional processing.
 */
static u32
pv_wait_head_or_lock(struct qspinlock *lock, struct qnode *node)
{
	struct qspinlock **lp = NULL;
	int waitcnt = 0;
	int loop;

	/*
	 * If pv_kick_node() already advanced our state, we don't need to
	 * insert ourselves into the hash table anymore.
	 */
	if (READ_ONCE(node->state) == vcpu_hashed)
		lp = (struct qspinlock **)1;

	/*
	 * Tracking # of slowpath locking operations
	 */
	lockevent_inc(lock_slowpath);

	for (;; waitcnt++) {
		/*
		 * Set correct vCPU state to be used by queue node wait-early
		 * mechanism.
		 */
		WRITE_ONCE(node->state, vcpu_running);

		/*
		 * Set the pending bit in the active lock spinning loop to
		 * disable lock stealing before attempting to acquire the lock.
		 */
		set_pending(lock);
		for (loop = SPIN_THRESHOLD; loop; loop--) {
			if (trylock_clear_pending(lock))
				goto gotlock;
			cpu_relax();
		}
		clear_pending(lock);


		if (!lp) { /* ONCE */
			lp = pv_hash(lock, node);

			/*
			 * We must hash before setting _Q_SLOW_VAL, such that
			 * when we observe _Q_SLOW_VAL in __pv_queued_spin_unlock()
			 * we'll be sure to be able to observe our hash entry.
			 *
			 *   [S] <hash>                 [Rmw] l->locked == _Q_SLOW_VAL
			 *       MB                           RMB
			 * [RmW] l->locked = _Q_SLOW_VAL  [L] <unhash>
			 *
			 * Matches the smp_rmb() in __pv_queued_spin_unlock().
			 */
			if (xchg(&lock->locked, _Q_SLOW_VAL) == 0) {
				/*
				 * The lock was free and now we own the lock.
				 * Change the lock value back to _Q_LOCKED_VAL
				 * and unhash the table.
				 */
				WRITE_ONCE(lock->locked, _Q_LOCKED_VAL);
				WRITE_ONCE(*lp, NULL);
				goto gotlock;
			}
		}
		WRITE_ONCE(node->state, vcpu_hashed);
		lockevent_inc(pv_wait_head);
		lockevent_cond_inc(pv_wait_again, waitcnt);
		pv_wait(&lock->locked, _Q_SLOW_VAL);

		/*
		 * Because of lock stealing, the queue head vCPU may not be
		 * able to acquire the lock before it has to wait again.
		 */
	}

	/*
	 * The cmpxchg() or xchg() call before coming here provides the
	 * acquire semantics for locking. The dummy ORing of _Q_LOCKED_VAL
	 * here is to indicate to the compiler that the value will always
	 * be nozero to enable better code optimization.
	 */
gotlock:
	return (u32)(atomic_read(&lock->val) | _Q_LOCKED_VAL);
}

/*
 * PV versions of the unlock fastpath and slowpath functions to be used
 * instead of queued_spin_unlock().
 */
__visible void
__pv_queued_spin_unlock_slowpath(struct qspinlock *lock, u8 locked)
{
	struct qnode *node;

	if (unlikely(locked != _Q_SLOW_VAL)) {
		WARN(!debug_locks_silent,
		     "pvqspinlock: lock 0x%lx has corrupted value 0x%x!\n",
		     (unsigned long)lock, atomic_read(&lock->val));
		return;
	}

	/*
	 * A failed cmpxchg doesn't provide any memory-ordering guarantees,
	 * so we need a barrier to order the read of the node data in
	 * pv_unhash *after* we've read the lock being _Q_SLOW_VAL.
	 *
	 * Matches the cmpxchg() in pv_wait_head_or_lock() setting _Q_SLOW_VAL.
	 */
	smp_rmb();

	/*
	 * Since the above failed to release, this must be the SLOW path.
	 * Therefore start by looking up the blocked node and unhashing it.
	 */
	node = pv_unhash(lock);

	/*
	 * Now that we have a reference to the (likely) blocked qnode,
	 * release the lock.
	 */
	smp_store_release(&lock->locked, 0);

	/*
	 * At this point the memory pointed at by lock can be freed/reused,
	 * however we can still use the qnode to kick the CPU.
	 * The other vCPU may not really be halted, but kicking an active
	 * vCPU is harmless other than the additional latency in completing
	 * the unlock.
	 */
	lockevent_inc(pv_kick_unlock);
	pv_kick(node->cpu);
}

#ifndef __pv_queued_spin_unlock
__visible void __pv_queued_spin_unlock(struct qspinlock *lock)
{
	u8 locked;

	/*
	 * We must not unlock if SLOW, because in that case we must first
	 * unhash. Otherwise it would be possible to have multiple @lock
	 * entries, which would be BAD.
	 */
	locked = cmpxchg_release(&lock->locked, _Q_LOCKED_VAL, 0);
	if (likely(locked == _Q_LOCKED_VAL))
		return;

	__pv_queued_spin_unlock_slowpath(lock, locked);
}
EXPORT_SYMBOL(__pv_queued_spin_unlock);
#endif

#else /* CONFIG_PARAVIRT_SPINLOCKS */
static __always_inline void pv_init_node(struct qnode *node) { }
static __always_inline void pv_wait_node(struct qnode *node,
					 struct qnode *prev) { }
static __always_inline void pv_kick_node(struct qspinlock *lock,
					 struct qnode *node) { }
static __always_inline u32  pv_wait_head_or_lock(struct qspinlock *lock,
						 struct qnode *node)
						   { return 0; }
static __always_inline bool pv_hybrid_queued_unfair_trylock(struct qspinlock *lock) { BUILD_BUG(); }
#endif /* CONFIG_PARAVIRT_SPINLOCKS */

static __always_inline void queued_spin_lock_mcs_queue(struct qspinlock *lock, bool paravirt)
{
	struct qnode *prev, *next, *node;
	u32 val, old, tail;
	int idx;

	BUILD_BUG_ON(CONFIG_NR_CPUS >= (1U << _Q_TAIL_CPU_BITS));

	node = this_cpu_ptr(&qnodes[0]);
	idx = node->count++;
	tail = encode_tail(smp_processor_id(), idx);

	trace_contention_begin(lock, LCB_F_SPIN);

	/*
	 * 4 nodes are allocated based on the assumption that there will
	 * not be nested NMIs taking spinlocks. That may not be true in
	 * some architectures even though the chance of needing more than
	 * 4 nodes will still be extremely unlikely. When that happens,
	 * we fall back to spinning on the lock directly without using
	 * any MCS node. This is not the most elegant solution, but is
	 * simple enough.
	 */
	if (unlikely(idx >= MAX_NODES)) {
		lockevent_inc(lock_no_node);
		if (paravirt) {
			while (!pv_hybrid_queued_unfair_trylock(lock))
				cpu_relax();
		} else {
			while (!queued_spin_trylock(lock))
				cpu_relax();
		}
		goto release;
	}

	node = grab_qnode(node, idx);

	/*
	 * Keep counts of non-zero index values:
	 */
	lockevent_cond_inc(lock_use_node2 + idx - 1, idx);

	/*
	 * Ensure that we increment the head node->count before initialising
	 * the actual node. If the compiler is kind enough to reorder these
	 * stores, then an IRQ could overwrite our assignments.
	 */
	barrier();

	node->locked = 0;
	node->next = NULL;
	if (paravirt)
		pv_init_node(node);

	/*
	 * We touched a (possibly) cold cacheline in the per-cpu queue node;
	 * attempt the trylock once more in the hope someone let go while we
	 * weren't watching.
	 */
	if (paravirt) {
		if (pv_hybrid_queued_unfair_trylock(lock))
			goto release;
	} else {
		if (queued_spin_trylock(lock))
			goto release;
	}

	/*
	 * Ensure that the initialisation of @node is complete before we
	 * publish the updated tail via xchg_tail() and potentially link
	 * @node into the waitqueue via WRITE_ONCE(prev->next, node) below.
	 */
	smp_wmb();

	/*
	 * Publish the updated tail.
	 * We have already touched the queueing cacheline; don't bother with
	 * pending stuff.
	 *
	 * p,*,* -> n,*,*
	 */
	old = xchg_tail(lock, tail);
	next = NULL;

	/*
	 * if there was a previous node; link it and wait until reaching the
	 * head of the waitqueue.
	 */
	if (old & _Q_TAIL_MASK) {
		prev = decode_tail(old);

		/* Link @node into the waitqueue. */
		WRITE_ONCE(prev->next, node);

		if (paravirt)
			pv_wait_node(node, prev);
		/* Wait for mcs node lock to be released */
		smp_cond_load_acquire(&node->locked, VAL);

		/*
		 * While waiting for the MCS lock, the next pointer may have
		 * been set by another lock waiter. We optimistically load
		 * the next pointer & prefetch the cacheline for writing
		 * to reduce latency in the upcoming MCS unlock operation.
		 */
		next = READ_ONCE(node->next);
		if (next)
			prefetchw(next);
	}

	/*
	 * we're at the head of the waitqueue, wait for the owner & pending to
	 * go away.
	 *
	 * *,x,y -> *,0,0
	 *
	 * this wait loop must use a load-acquire such that we match the
	 * store-release that clears the locked bit and create lock
	 * sequentiality; this is because the set_locked() function below
	 * does not imply a full barrier.
	 *
	 * The PV pv_wait_head_or_lock function, if active, will acquire
	 * the lock and return a non-zero value. So we have to skip the
	 * atomic_cond_read_acquire() call. As the next PV queue head hasn't
	 * been designated yet, there is no way for the locked value to become
	 * _Q_SLOW_VAL. So both the set_locked() and the
	 * atomic_cmpxchg_relaxed() calls will be safe.
	 *
	 * If PV isn't active, 0 will be returned instead.
	 *
	 */
	if (paravirt) {
		if ((val = pv_wait_head_or_lock(lock, node)))
			goto locked;
	}

	val = atomic_cond_read_acquire(&lock->val, !(VAL & _Q_LOCKED_PENDING_MASK));

locked:
	/*
	 * claim the lock:
	 *
	 * n,0,0 -> 0,0,1 : lock, uncontended
	 * *,*,0 -> *,*,1 : lock, contended
	 *
	 * If the queue head is the only one in the queue (lock value == tail)
	 * and nobody is pending, clear the tail code and grab the lock.
	 * Otherwise, we only need to grab the lock.
	 */

	/*
	 * In the PV case we might already have _Q_LOCKED_VAL set, because
	 * of lock stealing; therefore we must also allow:
	 *
	 * n,0,1 -> 0,0,1
	 *
	 * Note: at this point: (val & _Q_PENDING_MASK) == 0, because of the
	 *       above wait condition, therefore any concurrent setting of
	 *       PENDING will make the uncontended transition fail.
	 */
	if ((val & _Q_TAIL_MASK) == tail) {
		if (atomic_try_cmpxchg_relaxed(&lock->val, &val, _Q_LOCKED_VAL))
			goto release; /* No contention */
	}

	/*
	 * Either somebody is queued behind us or _Q_PENDING_VAL got set
	 * which will then detect the remaining tail and queue behind us
	 * ensuring we'll see a @next.
	 */
	set_locked(lock);

	/*
	 * contended path; wait for next if not observed yet, release.
	 */
	if (!next)
		next = smp_cond_load_relaxed(&node->next, (VAL));

	smp_store_release(&next->locked, 1); /* unlock the mcs node lock */
	if (paravirt)
		pv_kick_node(lock, next);

release:
	trace_contention_end(lock, 0);

	/*
	 * release the node
	 */
	__this_cpu_dec(qnodes[0].count);
}

/**
 * queued_spin_lock_slowpath - acquire the queued spinlock
 * @lock: Pointer to queued spinlock structure
 * @val: Current value of the queued spinlock 32-bit word
 *
 * (queue tail, pending bit, lock value)
 *
 *              fast     :    slow                                  :    unlock
 *                       :                                          :
 * uncontended  (0,0,0) -:--> (0,0,1) ------------------------------:--> (*,*,0)
 *                       :       | ^--------.------.             /  :
 *                       :       v           \      \            |  :
 * pending               :    (0,1,1) +--> (0,1,0)   \           |  :
 *                       :       | ^--'              |           |  :
 *                       :       v                   |           |  :
 * uncontended           :    (n,x,y) +--> (n,0,0) --'           |  :
 *   queue               :       | ^--'                          |  :
 *                       :       v                               |  :
 * contended             :    (*,x,y) +--> (*,0,0) ---> (*,0,1) -'  :
 *   queue               :         ^--'                             :
 */
void queued_spin_lock_slowpath(struct qspinlock *lock, u32 val)
{
	if (virt_spin_lock(lock))
		return;

	/*
	 * Wait for in-progress pending->locked hand-overs with a bounded
	 * number of spins so that we guarantee forward progress.
	 *
	 * 0,1,0 -> 0,0,1
	 */
	if (val == _Q_PENDING_VAL) {
		int cnt = _Q_PENDING_LOOPS;
		val = atomic_cond_read_relaxed(&lock->val,
					       (VAL != _Q_PENDING_VAL) || !cnt--);
	}

	/*
	 * If we observe any contention; queue.
	 */
	if (val & ~_Q_LOCKED_MASK)
		goto queue;

	/*
	 * trylock || pending
	 *
	 * 0,0,* -> 0,1,* -> 0,0,1 pending, trylock
	 */
	val = queued_fetch_set_pending_acquire(lock);

	/*
	 * If we observe contention, there is a concurrent locker.
	 *
	 * Undo and queue; our setting of PENDING might have made the
	 * n,0,0 -> 0,0,0 transition fail and it will now be waiting
	 * on @next to become !NULL.
	 */
	if (unlikely(val & ~_Q_LOCKED_MASK)) {

		/* Undo PENDING if we set it. */
		if (!(val & _Q_PENDING_MASK))
			clear_pending(lock);

		goto queue;
	}

	/*
	 * We're pending, wait for the owner to go away.
	 *
	 * 0,1,1 -> 0,1,0
	 *
	 * this wait loop must be a load-acquire such that we match the
	 * store-release that clears the locked bit and create lock
	 * sequentiality; this is because not all
	 * clear_pending_set_locked() implementations imply full
	 * barriers.
	 */
	if (val & _Q_LOCKED_MASK)
		atomic_cond_read_acquire(&lock->val, !(VAL & _Q_LOCKED_MASK));

	/*
	 * take ownership and clear the pending bit.
	 *
	 * 0,1,0 -> 0,0,1
	 */
	clear_pending_set_locked(lock);
	lockevent_inc(lock_pending);
	return;

	/*
	 * End of pending bit optimistic spinning and beginning of MCS
	 * queuing.
	 */
queue:
	lockevent_inc(lock_slowpath);
	queued_spin_lock_mcs_queue(lock, false);
}
EXPORT_SYMBOL(queued_spin_lock_slowpath);

#ifdef CONFIG_PARAVIRT_SPINLOCKS
void __pv_queued_spin_lock_slowpath(struct qspinlock *lock, u32 val)
{
	queued_spin_lock_mcs_queue(lock, true);
}
EXPORT_SYMBOL(__pv_queued_spin_lock_slowpath);

bool nopvspin __initdata;
static __init int parse_nopvspin(char *arg)
{
	nopvspin = true;
	return 0;
}
early_param("nopvspin", parse_nopvspin);
#endif /* CONFIG_PARAVIRT_SPINLOCKS */
