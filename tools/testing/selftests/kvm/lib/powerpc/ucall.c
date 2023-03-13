// SPDX-License-Identifier: GPL-2.0
/*
 * ucall support. A ucall is a "hypercall to host userspace".
 */
#include "kvm_util.h"
#include "hcall.h"

void ucall_arch_init(struct kvm_vm *vm, vm_paddr_t mmio_gpa)
{
}

void ucall_arch_do_ucall(vm_vaddr_t uc)
{
	hcall2(H_UCALL, UCALL_R4_UCALL, (uintptr_t)(uc));
}

void *ucall_arch_get_ucall(struct kvm_vcpu *vcpu)
{
	struct kvm_run *run = vcpu->run;

	if (run->exit_reason == KVM_EXIT_PAPR_HCALL &&
	    run->papr_hcall.nr == H_UCALL) {
		struct kvm_regs regs;

		vcpu_regs_get(vcpu, &regs);
		if (regs.gpr[4] == UCALL_R4_UCALL)
			return (void *)regs.gpr[5];
	}
	return NULL;
}
