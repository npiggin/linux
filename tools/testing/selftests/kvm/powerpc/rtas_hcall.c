// SPDX-License-Identifier: GPL-2.0-only
/*
 * Test the KVM H_RTAS hcall and copying buffers between guest and host.
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
#include "hcall.h"

struct rtas_args {
	__be32 token;
	__be32 nargs;
	__be32 nret;
	__be32 args[16];
        __be32 *rets;     /* Pointer to return values in args[]. */
};

static void guest_code(void)
{
	struct rtas_args r;
	int64_t rc;

	r.token = cpu_to_be32(0xdeadbeef);
	r.nargs = cpu_to_be32(3);
	r.nret = cpu_to_be32(2);
	r.rets = &r.args[3];
	r.args[0] = cpu_to_be32(0x1000);
	r.args[1] = cpu_to_be32(0x1001);
	r.args[2] = cpu_to_be32(0x1002);
	rc = hcall1(H_RTAS, (uint64_t)&r);
	GUEST_ASSERT(rc == 0);
	GUEST_ASSERT_1(be32_to_cpu(r.rets[0]) == 0xabc, be32_to_cpu(r.rets[0]));
	GUEST_ASSERT_1(be32_to_cpu(r.rets[1]) == 0x123, be32_to_cpu(r.rets[1]));

	GUEST_DONE();
}

int main(int argc, char *argv[])
{
	struct kvm_regs regs;
	struct rtas_args *r;
	vm_vaddr_t rtas_vaddr;
	struct ucall uc;
	struct kvm_vcpu *vcpu;
	struct kvm_vm *vm;
	uint64_t tmp;
	int ret;

	ksft_print_header();

	ksft_set_plan(1);

	/* Create VM */
	vm = vm_create_with_one_vcpu(&vcpu, guest_code);

	printf("Running H_RTAS guest vcpu.\n");

	ret = _vcpu_run(vcpu);
	TEST_ASSERT(ret == 0, "vcpu_run failed: %d\n", ret);
	switch ((tmp = get_ucall(vcpu, &uc))) {
	case UCALL_NONE:
		break; // good
	case UCALL_DONE:
		TEST_FAIL("Unexpected final guest exit %lu\n", tmp);
		break;
	case UCALL_ABORT:
		REPORT_GUEST_ASSERT_N(uc, "values: %lu (0x%lx)\n",
				      GUEST_ASSERT_ARG(uc, 0),
				      GUEST_ASSERT_ARG(uc, 0));
		break;
	default:
		TEST_FAIL("Unexpected guest exit %lu\n", tmp);
	}

	TEST_ASSERT(vcpu->run->exit_reason == KVM_EXIT_PAPR_HCALL,
		    "Expected PAPR_HCALL exit, got %s\n",
		    exit_reason_str(vcpu->run->exit_reason));
	TEST_ASSERT(vcpu->run->papr_hcall.nr == H_RTAS,
		    "Expected H_RTAS exit, got %lld\n",
		    vcpu->run->papr_hcall.nr);

	printf("Got H_RTAS exit.\n");

	vcpu_regs_get(vcpu, &regs);
	rtas_vaddr = regs.gpr[4];
	printf("H_RTAS rtas_args at gEA=0x%lx\n", rtas_vaddr);

	r = addr_gva2hva(vm, rtas_vaddr);

	TEST_ASSERT(r->token == cpu_to_be32(0xdeadbeef),
		    "Expected RTAS token 0xdeadbeef, got 0x%x\n",
		    be32_to_cpu(r->token));
	TEST_ASSERT(r->nargs == cpu_to_be32(3),
		    "Expected RTAS nargs 3, got %u\n",
		    be32_to_cpu(r->nargs));
	TEST_ASSERT(r->nret == cpu_to_be32(2),
		    "Expected RTAS nret 2, got %u\n",
		    be32_to_cpu(r->nret));
	TEST_ASSERT(r->args[0] == cpu_to_be32(0x1000),
		    "Expected args[0] to be 0x1000, got 0x%x\n",
		    be32_to_cpu(r->args[0]));
	TEST_ASSERT(r->args[1] == cpu_to_be32(0x1001),
		    "Expected args[1] to be 0x1001, got 0x%x\n",
		    be32_to_cpu(r->args[1]));
	TEST_ASSERT(r->args[2] == cpu_to_be32(0x1002),
		    "Expected args[2] to be 0x1002, got 0x%x\n",
		    be32_to_cpu(r->args[2]));

	printf("Guest rtas_args is correct, setting rets.\n");

	r->args[3] = cpu_to_be32(0xabc);
	r->args[4] = cpu_to_be32(0x123);

	regs.gpr[3] = 0;
	vcpu_regs_set(vcpu, &regs);

	printf("Running H_RTAS guest vcpu again (hcall return H_SUCCESS).\n");

	ret = _vcpu_run(vcpu);
	TEST_ASSERT(ret == 0, "vcpu_run failed: %d\n", ret);
	switch ((tmp = get_ucall(vcpu, &uc))) {
	case UCALL_DONE:
		printf("Got final guest exit.\n");
		break;
	case UCALL_ABORT:
		REPORT_GUEST_ASSERT_N(uc, "values: %lu (0x%lx)\n",
				      GUEST_ASSERT_ARG(uc, 0),
				      GUEST_ASSERT_ARG(uc, 0));
		break;
	default:
		TEST_FAIL("Unexpected guest exit %lu\n", tmp);
	}

	kvm_vm_free(vm);

	ksft_test_result_pass("%s\n", "rtas test");
	ksft_finished();	/* Print results and exit() accordingly */
}
