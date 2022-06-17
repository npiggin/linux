// SPDX-License-Identifier: GPL-2.0
/*
 * Split spinlock implementation out into its own file, so it can be
 * compiled in a FTRACE-compatible way.
 */
#include <linux/spinlock.h>
#include <linux/export.h>
#include <linux/jump_label.h>

#include <asm/ibt.h>
#include <asm/paravirt.h>

__visible void __native_queued_spin_unlock(struct qspinlock *lock)
{
	native_queued_spin_unlock(lock);
}
PV_CALLEE_SAVE_REGS_THUNK(__native_queued_spin_unlock);

#ifdef CONFIG_PARAVIRT_SPINLOCKS
/*
 * For x86-64, PV_CALLEE_SAVE_REGS_THUNK() saves and restores 8 64-bit
 * registers. For i386, however, only 1 32-bit register needs to be saved
 * and restored. So an optimized version of __pv_queued_spin_unlock() is
 * hand-coded for 64-bit, but it isn't worthwhile to do it for 32-bit.
 */
#ifdef CONFIG_64BIT

__visible void
__pv_queued_spin_unlock_slowpath(struct qspinlock *lock, u8 locked);

PV_CALLEE_SAVE_REGS_THUNK(__pv_queued_spin_unlock_slowpath);
#define PV_UNLOCK		"__raw_callee_save___pv_queued_spin_unlock"
#define PV_UNLOCK_SLOWPATH	"__raw_callee_save___pv_queued_spin_unlock_slowpath"

/*
 * Optimized assembly version of __raw_callee_save___pv_queued_spin_unlock
 * which combines the registers saving trunk and the body of the following
 * C code:
 *
 * void __pv_queued_spin_unlock(struct qspinlock *lock)
 * {
 *	u8 lockval = cmpxchg(&lock->locked, _Q_LOCKED_VAL, 0);
 *
 *	if (likely(lockval == _Q_LOCKED_VAL))
 *		return;
 *	pv_queued_spin_unlock_slowpath(lock, lockval);
 * }
 *
 * For x86-64,
 *   rdi = lock              (first argument)
 *   rsi = lockval           (second argument)
 *   rdx = internal variable (set to 0)
 */
asm    (".pushsection .text;"
	".globl " PV_UNLOCK ";"
	".type " PV_UNLOCK ", @function;"
	".align 4,0x90;"
	PV_UNLOCK ": "
	ASM_ENDBR
	FRAME_BEGIN
	"push  %rdx;"
	"mov   $0x1,%eax;"
	"xor   %edx,%edx;"
	LOCK_PREFIX "cmpxchg %dl,(%rdi);"
	"cmp   $0x1,%al;"
	"jne   .slowpath;"
	"pop   %rdx;"
	FRAME_END
	ASM_RET
	".slowpath: "
	"push   %rsi;"
	"movzbl %al,%esi;"
	"call " PV_UNLOCK_SLOWPATH ";"
	"pop    %rsi;"
	"pop    %rdx;"
	FRAME_END
	ASM_RET
	".size " PV_UNLOCK ", .-" PV_UNLOCK ";"
	".popsection");

#else /* CONFIG_64BIT */

extern void __pv_queued_spin_unlock(struct qspinlock *lock);
PV_CALLEE_SAVE_REGS_THUNK(__pv_queued_spin_unlock);

#endif /* CONFIG_64BIT */
#endif /* CONFIG_PARAVIRT_SPINLOCKS */

bool pv_is_native_spin_unlock(void)
{
	return pv_ops.lock.queued_spin_unlock.func ==
		__raw_callee_save___native_queued_spin_unlock;
}

__visible bool __native_vcpu_is_preempted(long cpu)
{
	return false;
}
PV_CALLEE_SAVE_REGS_THUNK(__native_vcpu_is_preempted);

bool pv_is_native_vcpu_is_preempted(void)
{
	return pv_ops.lock.vcpu_is_preempted.func ==
		__raw_callee_save___native_vcpu_is_preempted;
}

void __init paravirt_set_cap(void)
{
	if (!pv_is_native_spin_unlock())
		setup_force_cpu_cap(X86_FEATURE_PVUNLOCK);

	if (!pv_is_native_vcpu_is_preempted())
		setup_force_cpu_cap(X86_FEATURE_VCPUPREEMPT);
}
