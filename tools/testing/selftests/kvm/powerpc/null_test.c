// SPDX-License-Identifier: GPL-2.0-only
/*
 * Tests for guest creation, run, ucall, interrupt, and vm dumping.
 */

#define _GNU_SOURCE /* for program_invocation_short_name */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>

#include "test_util.h"
#include "kvm_util.h"
#include "kselftest.h"
#include "processor.h"
#include "helpers.h"

extern void guest_code_asm(void);
asm(".global guest_code_asm");
asm(".balign 4");
asm("guest_code_asm:");
asm("li 3,0"); // H_UCALL
asm("li 4,0"); // UCALL_R4_SIMPLE
asm("sc 1");

static void test_asm(void)
{
	struct kvm_vcpu *vcpu;
	struct kvm_vm *vm;

	vm = vm_create_with_one_vcpu(&vcpu, guest_code_asm);

	vcpu_run(vcpu);
	handle_ucall(vcpu, UCALL_NONE);

	kvm_vm_free(vm);
}

static void guest_code_ucall(void)
{
	GUEST_DONE();
}

static void test_ucall(void)
{
	struct kvm_vcpu *vcpu;
	struct kvm_vm *vm;

	vm = vm_create_with_one_vcpu(&vcpu, guest_code_ucall);

	vcpu_run(vcpu);
	handle_ucall(vcpu, UCALL_DONE);

	kvm_vm_free(vm);
}

static void trap_handler(struct ex_regs *regs)
{
	GUEST_SYNC(1);
	regs->nia += 4;
}

static void guest_code_trap(void)
{
	GUEST_SYNC(0);
	asm volatile("trap");
	GUEST_DONE();
}

static void test_trap(void)
{
	struct kvm_vcpu *vcpu;
	struct kvm_vm *vm;

	vm = vm_create_with_one_vcpu(&vcpu, guest_code_trap);
	vm_install_exception_handler(vm, 0x700, trap_handler);

	vcpu_run(vcpu);
	host_sync(vcpu, 0);
	vcpu_run(vcpu);
	host_sync(vcpu, 1);
	vcpu_run(vcpu);
	handle_ucall(vcpu, UCALL_DONE);

	vm_install_exception_handler(vm, 0x700, NULL);

	kvm_vm_free(vm);
}

static void dsi_handler(struct ex_regs *regs)
{
	GUEST_SYNC(1);
	regs->nia += 4;
}

static void guest_code_dsi(void)
{
	GUEST_SYNC(0);
	asm volatile("stb %r0,0(0)");
	GUEST_DONE();
}

static void test_dsi(void)
{
	struct kvm_vcpu *vcpu;
	struct kvm_vm *vm;

	vm = vm_create_with_one_vcpu(&vcpu, guest_code_dsi);
	vm_install_exception_handler(vm, 0x300, dsi_handler);

	vcpu_run(vcpu);
	host_sync(vcpu, 0);
	vcpu_run(vcpu);
	host_sync(vcpu, 1);
	vcpu_run(vcpu);
	handle_ucall(vcpu, UCALL_DONE);

	vm_install_exception_handler(vm, 0x300, NULL);

	kvm_vm_free(vm);
}

static void test_dump(void)
{
	struct kvm_vcpu *vcpu;
	struct kvm_vm *vm;

	vm = vm_create_with_one_vcpu(&vcpu, guest_code_ucall);

	vcpu_run(vcpu);
	handle_ucall(vcpu, UCALL_DONE);

	printf("Testing vm_dump...\n");
	vm_dump(stderr, vm, 2);

	kvm_vm_free(vm);
}


struct testdef {
	const char *name;
	void (*test)(void);
} testlist[] = {
	{ "null asm test", test_asm},
	{ "null ucall test", test_ucall},
	{ "trap test", test_trap},
	{ "page fault test", test_dsi},
	{ "vm dump test", test_dump},
};

int main(int argc, char *argv[])
{
	int idx;

	ksft_print_header();

	ksft_set_plan(ARRAY_SIZE(testlist));

	for (idx = 0; idx < ARRAY_SIZE(testlist); idx++) {
		testlist[idx].test();
		ksft_test_result_pass("%s\n", testlist[idx].name);
	}

	ksft_finished();	/* Print results and exit() accordingly */
}
