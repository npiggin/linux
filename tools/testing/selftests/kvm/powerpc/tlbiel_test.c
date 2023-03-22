// SPDX-License-Identifier: GPL-2.0-only
/*
 * Test TLBIEL virtualisation, TLBIEL should appear to invalidate TLBs on the
 * vCPU, despite moving between physical CPUs.
 */

#define _GNU_SOURCE /* for program_invocation_short_name */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <signal.h>

#include "test_util.h"
#include "kvm_util.h"
#include "kselftest.h"
#include "processor.h"
#include "helpers.h"

static int nr_cpus;
static int *cpu_array;

static void set_cpu(int cpu)
{
	cpu_set_t set;

	CPU_ZERO(&set);
	CPU_SET(cpu, &set);

	if (sched_setaffinity(0, sizeof(set), &set) == -1) {
		perror("sched_setaffinity");
		exit(1);
	}
}

static void set_random_cpu(void)
{
	set_cpu(cpu_array[random() % nr_cpus]);
}

static void init_sched_cpu(void)
{
	cpu_set_t possible_mask;
	int i, cnt, nproc;

	nproc = get_nprocs_conf();

	TEST_ASSERT(!sched_getaffinity(0, sizeof(possible_mask), &possible_mask),
		"sched_getaffinity failed, errno = %d (%s)", errno, strerror(errno));

	nr_cpus = CPU_COUNT(&possible_mask);
	cpu_array = malloc(nr_cpus * sizeof(int));

	cnt = 0;
	for (i = 0; i < nproc; i++) {
		if (CPU_ISSET(i, &possible_mask)) {
			cpu_array[cnt] = i;
			cnt++;
		}
	}
}

static volatile bool timeout;

static void set_timer(int sec)
{
	struct itimerval timer;

	timeout = false;

	timer.it_value.tv_sec  = sec;
	timer.it_value.tv_usec = 0;
	timer.it_interval = timer.it_value;
	TEST_ASSERT(setitimer(ITIMER_REAL, &timer, NULL) == 0,
			"setitimer failed %s", strerror(errno));
}

static void sigalrm_handler(int sig)
{
	timeout = true;
}

static void init_timers(void)
{
	TEST_ASSERT(signal(SIGALRM, sigalrm_handler) != SIG_ERR,
		    "Failed to register SIGALRM handler, errno = %d (%s)",
		    errno, strerror(errno));
}

static inline void virt_invalidate_tlb(uint64_t gva)
{
	unsigned long rb, rs;
	unsigned long is = 2, ric = 0, prs = 1, r = 1;

	rb = is << 10;
	rs = 0;

	asm volatile("ptesync ; .machine push ; .machine power9 ; tlbiel %0,%1,%2,%3,%4 ; .machine pop ; ptesync"
			:: "r"(rb), "r"(rs), "i"(ric), "i"(prs), "i"(r)
			: "memory");
}

static inline void virt_invalidate_pwc(uint64_t gva)
{
	unsigned long rb, rs;
	unsigned long is = 2, ric = 1, prs = 1, r = 1;

	rb = is << 10;
	rs = 0;

	asm volatile("ptesync ; .machine push ; .machine power9 ; tlbiel %0,%1,%2,%3,%4 ; .machine pop ; ptesync"
			:: "r"(rb), "r"(rs), "i"(ric), "i"(prs), "i"(r)
			: "memory");
}

static inline void virt_invalidate_all(uint64_t gva)
{
	unsigned long rb, rs;
	unsigned long is = 2, ric = 2, prs = 1, r = 1;

	rb = is << 10;
	rs = 0;

	asm volatile("ptesync ; .machine push ; .machine power9 ; tlbiel %0,%1,%2,%3,%4 ; .machine pop ; ptesync"
			:: "r"(rb), "r"(rs), "i"(ric), "i"(prs), "i"(r)
			: "memory");
}

static inline void virt_invalidate_page(uint64_t gva)
{
	unsigned long rb, rs;
	unsigned long is = 0, ric = 0, prs = 1, r = 1;
	unsigned long ap = 0x5;
	unsigned long epn = gva & ~0xffffUL;
	unsigned long pid = 0;

	rb = epn | (is << 10) | (ap << 5);
	rs = pid << 32;

	asm volatile("ptesync ; .machine push ; .machine power9 ; tlbiel %0,%1,%2,%3,%4 ; .machine pop ; ptesync"
			:: "r"(rb), "r"(rs), "i"(ric), "i"(prs), "i"(r)
			: "memory");
}

enum {
	SYNC_BEFORE_LOAD1,
	SYNC_BEFORE_LOAD2,
	SYNC_BEFORE_STORE,
	SYNC_BEFORE_INVALIDATE,
	SYNC_DSI,
};

static void remap_dsi_handler(struct ex_regs *regs)
{
	GUEST_ASSERT(0);
}

#define PAGE1_VAL 0x1234567890abcdef
#define PAGE2_VAL 0x5c5c5c5c5c5c5c5c

static void remap_guest_code(vm_vaddr_t page)
{
	unsigned long *mem = (void *)page;

	for (;;) {
		unsigned long tmp;

		GUEST_SYNC(SYNC_BEFORE_LOAD1);
		asm volatile("ld %0,%1" : "=r"(tmp) : "m"(*mem));
		GUEST_ASSERT(tmp == PAGE1_VAL);
		GUEST_SYNC(SYNC_BEFORE_INVALIDATE);
		virt_invalidate_page(page);
		GUEST_SYNC(SYNC_BEFORE_LOAD2);
		asm volatile("ld %0,%1" : "=r"(tmp) : "m"(*mem));
		GUEST_ASSERT(tmp == PAGE2_VAL);
		GUEST_SYNC(SYNC_BEFORE_INVALIDATE);
		virt_invalidate_page(page);
	}
}

static void remap_test(void)
{
	struct kvm_vcpu *vcpu;
	struct kvm_vm *vm;
	vm_vaddr_t vaddr;
	vm_paddr_t pages[2];
	uint64_t *hostptr;

	/* Create VM */
	vm = vm_create_with_one_vcpu(&vcpu, remap_guest_code);
	vm_install_exception_handler(vm, 0x300, remap_dsi_handler);

	vaddr = vm_vaddr_alloc_page(vm);
	pages[0] = addr_gva2gpa(vm, vaddr);
	pages[1] = vm_phy_page_alloc(vm, 0, vm->memslots[MEM_REGION_DATA]);

	hostptr = addr_gpa2hva(vm, pages[0]);
	*hostptr = PAGE1_VAL;

	hostptr = addr_gpa2hva(vm, pages[1]);
	*hostptr = PAGE2_VAL;

	vcpu_args_set(vcpu, 1, vaddr);

	set_random_cpu();
	set_timer(10);

	while (!timeout) {
		vcpu_run(vcpu);

		host_sync(vcpu, SYNC_BEFORE_LOAD1);
		set_random_cpu();
		vcpu_run(vcpu);

		host_sync(vcpu, SYNC_BEFORE_INVALIDATE);
		set_random_cpu();
		TEST_ASSERT(virt_remap_pte(vm, vaddr, pages[1]), "Remap page1 failed");
		vcpu_run(vcpu);

		host_sync(vcpu, SYNC_BEFORE_LOAD2);
		set_random_cpu();
		vcpu_run(vcpu);

		host_sync(vcpu, SYNC_BEFORE_INVALIDATE);
		TEST_ASSERT(virt_remap_pte(vm, vaddr, pages[0]), "Remap page0 failed");
		set_random_cpu();
	}

	vm_install_exception_handler(vm, 0x300, NULL);

	kvm_vm_free(vm);
}

static void wrprotect_dsi_handler(struct ex_regs *regs)
{
	GUEST_SYNC(SYNC_DSI);
	regs->nia += 4;
}

static void wrprotect_guest_code(vm_vaddr_t page)
{
	volatile char *mem = (void *)page;

	for (;;) {
		GUEST_SYNC(SYNC_BEFORE_STORE);
		asm volatile("stb %1,%0" : "=m"(*mem) : "r"(1));
		GUEST_SYNC(SYNC_BEFORE_INVALIDATE);
		virt_invalidate_page(page);
	}
}

static void wrprotect_test(void)
{
	struct kvm_vcpu *vcpu;
	struct kvm_vm *vm;
	vm_vaddr_t page;
	void *hostptr;

	/* Create VM */
	vm = vm_create_with_one_vcpu(&vcpu, wrprotect_guest_code);
	vm_install_exception_handler(vm, 0x300, wrprotect_dsi_handler);

	page = vm_vaddr_alloc_page(vm);
	hostptr = addr_gva2hva(vm, page);
	memset(hostptr, 0, vm->page_size);

	vcpu_args_set(vcpu, 1, page);

	set_random_cpu();
	set_timer(10);

	while (!timeout) {
		vcpu_run(vcpu);
		host_sync(vcpu, SYNC_BEFORE_STORE);

		vcpu_run(vcpu);
		host_sync(vcpu, SYNC_BEFORE_INVALIDATE);

		TEST_ASSERT(virt_wrprotect_pte(vm, page), "Wrprotect page failed");
		/* Invalidate on different CPU */
		set_random_cpu();
		vcpu_run(vcpu);
		host_sync(vcpu, SYNC_BEFORE_STORE);

		/* Store on different CPU */
		set_random_cpu();
		vcpu_run(vcpu);
		host_sync(vcpu, SYNC_DSI);
		vcpu_run(vcpu);
		host_sync(vcpu, SYNC_BEFORE_INVALIDATE);

		TEST_ASSERT(virt_wrenable_pte(vm, page), "Wrenable page failed");

		/* Invalidate on different CPU when we go around */
		set_random_cpu();
	}

	vm_install_exception_handler(vm, 0x300, NULL);

	kvm_vm_free(vm);
}

static void wrp_mt_dsi_handler(struct ex_regs *regs)
{
	GUEST_SYNC(SYNC_DSI);
	regs->nia += 4;
}

static void wrp_mt_guest_code(vm_vaddr_t page, bool invalidates)
{
	volatile char *mem = (void *)page;

	for (;;) {
		GUEST_SYNC(SYNC_BEFORE_STORE);
		asm volatile("stb %1,%0" : "=m"(*mem) : "r"(1));
		if (invalidates) {
			GUEST_SYNC(SYNC_BEFORE_INVALIDATE);
			virt_invalidate_page(page);
		}
	}
}

static void wrp_mt_test(void)
{
	struct kvm_vcpu *vcpu[2];
	struct kvm_vm *vm;
	vm_vaddr_t page;
	void *hostptr;

	/* Create VM */
	vm = vm_create_with_vcpus(2, wrp_mt_guest_code, vcpu);
	vm_install_exception_handler(vm, 0x300, wrp_mt_dsi_handler);

	page = vm_vaddr_alloc_page(vm);
	hostptr = addr_gva2hva(vm, page);
	memset(hostptr, 0, vm->page_size);

	vcpu_args_set(vcpu[0], 2, page, 1);
	vcpu_args_set(vcpu[1], 2, page, 0);

	set_random_cpu();
	set_timer(10);

	while (!timeout) {
		/* Run vcpu[1] only when page is writable, should never fault */
		vcpu_run(vcpu[1]);
		host_sync(vcpu[1], SYNC_BEFORE_STORE);

		vcpu_run(vcpu[0]);
		host_sync(vcpu[0], SYNC_BEFORE_STORE);

		vcpu_run(vcpu[0]);
		host_sync(vcpu[0], SYNC_BEFORE_INVALIDATE);

		TEST_ASSERT(virt_wrprotect_pte(vm, page), "Wrprotect page failed");
		/* Invalidate on different CPU */
		set_random_cpu();
		vcpu_run(vcpu[0]);
		host_sync(vcpu[0], SYNC_BEFORE_STORE);

		/* Store on different CPU */
		set_random_cpu();
		vcpu_run(vcpu[0]);
		host_sync(vcpu[0], SYNC_DSI);
		vcpu_run(vcpu[0]);
		host_sync(vcpu[0], SYNC_BEFORE_INVALIDATE);

		TEST_ASSERT(virt_wrenable_pte(vm, page), "Wrenable page failed");
		/* Invalidate on different CPU when we go around */
		set_random_cpu();
	}

	vm_install_exception_handler(vm, 0x300, NULL);

	kvm_vm_free(vm);
}

static void proctbl_dsi_handler(struct ex_regs *regs)
{
	GUEST_SYNC(SYNC_DSI);
	regs->nia += 4;
}

static void proctbl_guest_code(vm_vaddr_t page)
{
	volatile char *mem = (void *)page;

	for (;;) {
		GUEST_SYNC(SYNC_BEFORE_STORE);
		asm volatile("stb %1,%0" : "=m"(*mem) : "r"(1));
		GUEST_SYNC(SYNC_BEFORE_INVALIDATE);
		virt_invalidate_all(page);
	}
}

static void proctbl_test(void)
{
	struct kvm_vcpu *vcpu;
	struct kvm_vm *vm;
	vm_vaddr_t page;
	vm_paddr_t orig_pgd;
	vm_paddr_t alternate_pgd;
	void *hostptr;

	/* Create VM */
	vm = vm_create_with_one_vcpu(&vcpu, proctbl_guest_code);
	vm_install_exception_handler(vm, 0x300, proctbl_dsi_handler);

	page = vm_vaddr_alloc_page(vm);
	hostptr = addr_gva2hva(vm, page);
	memset(hostptr, 0, vm->page_size);

	orig_pgd = vm->pgd;
	alternate_pgd = virt_pt_duplicate(vm);

	/* Write protect the original PTE */
	TEST_ASSERT(virt_wrprotect_pte(vm, page), "Wrprotect page failed");

	vm->pgd = alternate_pgd;
	set_radix_proc_table(vm, 0, vm->pgd);

	vcpu_args_set(vcpu, 1, page);

	set_random_cpu();
	set_timer(10);

	while (!timeout) {
		vcpu_run(vcpu);
		host_sync(vcpu, SYNC_BEFORE_STORE);

		vcpu_run(vcpu);
		host_sync(vcpu, SYNC_BEFORE_INVALIDATE);
		/* Writeable store succeeds */

		/* Swap page tables to write protected one */
		vm->pgd = orig_pgd;
		set_radix_proc_table(vm, 0, vm->pgd);

		/* Invalidate on different CPU */
		set_random_cpu();
		vcpu_run(vcpu);
		host_sync(vcpu, SYNC_BEFORE_STORE);

		/* Store on different CPU */
		set_random_cpu();
		vcpu_run(vcpu);
		host_sync(vcpu, SYNC_DSI);
		vcpu_run(vcpu);
		host_sync(vcpu, SYNC_BEFORE_INVALIDATE);

		/* Swap page tables to write enabled one */
		vm->pgd = alternate_pgd;
		set_radix_proc_table(vm, 0, vm->pgd);

		/* Invalidate on different CPU when we go around */
		set_random_cpu();
	}
	vm->pgd = orig_pgd;
	set_radix_proc_table(vm, 0, vm->pgd);

	vm_install_exception_handler(vm, 0x300, NULL);

	kvm_vm_free(vm);
}

struct testdef {
	const char *name;
	void (*test)(void);
} testlist[] = {
	{ "tlbiel wrprotect test", wrprotect_test},
	{ "tlbiel wrprotect 2-vCPU test", wrp_mt_test},
	{ "tlbiel process table update test", proctbl_test},
	{ "tlbiel remap test", remap_test},
};

int main(int argc, char *argv[])
{
	int idx;

	ksft_print_header();

	ksft_set_plan(ARRAY_SIZE(testlist));

	init_sched_cpu();
	init_timers();

	for (idx = 0; idx < ARRAY_SIZE(testlist); idx++) {
		testlist[idx].test();
		ksft_test_result_pass("%s\n", testlist[idx].name);
	}

	ksft_finished();	/* Print results and exit() accordingly */
}
