// SPDX-License-Identifier: GPL-2.0-only
#include <linux/kernel.h>
#include <linux/kvm_host.h>
#include <asm/asm-prototypes.h>
#include <asm/dbell.h>
#include <asm/kvm_ppc.h>
#include <asm/pmc.h>
#include <asm/ppc-opcode.h>

#include "book3s_hv.h"

static void load_spr_state(struct kvm_vcpu *vcpu,
			   struct p9_host_os_sprs *host_os_sprs)
{
	/* TAR is very fast */
	mtspr(SPRN_TAR, vcpu->arch.tar);

#ifdef CONFIG_ALTIVEC
	if (cpu_has_feature(CPU_FTR_ALTIVEC) &&
			current->thread.vrsave != vcpu->arch.vrsave)
		mtspr(SPRN_VRSAVE, vcpu->arch.vrsave);
#endif

	if (vcpu->arch.hfscr & HFSCR_EBB) {
		if (current->thread.ebbhr != vcpu->arch.ebbhr)
			mtspr(SPRN_EBBHR, vcpu->arch.ebbhr);
		if (current->thread.ebbrr != vcpu->arch.ebbrr)
			mtspr(SPRN_EBBRR, vcpu->arch.ebbrr);
		if (current->thread.bescr != vcpu->arch.bescr)
			mtspr(SPRN_BESCR, vcpu->arch.bescr);
	}

	if (!cpu_has_feature(CPU_FTR_ARCH_31) && current->thread.tidr != vcpu->arch.tid)
		mtspr(SPRN_TIDR, vcpu->arch.tid);
	if (host_os_sprs->iamr != vcpu->arch.iamr)
		mtspr(SPRN_IAMR, vcpu->arch.iamr);
	if (host_os_sprs->amr != vcpu->arch.amr)
		mtspr(SPRN_AMR, vcpu->arch.amr);
	if (vcpu->arch.uamor != 0)
		mtspr(SPRN_UAMOR, vcpu->arch.uamor);
	if (host_os_sprs->fscr != vcpu->arch.fscr)
		mtspr(SPRN_FSCR, vcpu->arch.fscr);
	if (host_os_sprs->dscr != vcpu->arch.dscr)
		mtspr(SPRN_DSCR, vcpu->arch.dscr);
	if (vcpu->arch.pspb != 0)
		mtspr(SPRN_PSPB, vcpu->arch.pspb);

	/*
	 * DAR, DSISR, and for nested HV, SPRGs must be set with MSR[RI]
	 * clear (or hstate set appropriately to catch those registers
	 * being clobbered if we take a MCE or SRESET), so those are done
	 * later.
	 */

	if (!(vcpu->arch.ctrl & 1))
		mtspr(SPRN_CTRLT, 0);
}

static void store_spr_state(struct kvm_vcpu *vcpu)
{
	vcpu->arch.tar = mfspr(SPRN_TAR);

	if (vcpu->arch.hfscr & HFSCR_EBB) {
		vcpu->arch.ebbhr = mfspr(SPRN_EBBHR);
		vcpu->arch.ebbrr = mfspr(SPRN_EBBRR);
		vcpu->arch.bescr = mfspr(SPRN_BESCR);
		vcpu->arch.hfscr &= ~HFSCR_EBB;
	}

	if (!cpu_has_feature(CPU_FTR_ARCH_31))
		vcpu->arch.tid = mfspr(SPRN_TIDR);
	vcpu->arch.iamr = mfspr(SPRN_IAMR);
	vcpu->arch.amr = mfspr(SPRN_AMR);
	vcpu->arch.uamor = mfspr(SPRN_UAMOR);
	vcpu->arch.fscr = mfspr(SPRN_FSCR);
	vcpu->arch.dscr = mfspr(SPRN_DSCR);
	vcpu->arch.pspb = mfspr(SPRN_PSPB);

	vcpu->arch.ctrl = mfspr(SPRN_CTRLF);
}

void load_vcpu_state(struct kvm_vcpu *vcpu,
			   struct p9_host_os_sprs *host_os_sprs)
{
	if ((cpu_has_feature(CPU_FTR_TM) ||
	    cpu_has_feature(CPU_FTR_P9_TM_HV_ASSIST)) &&
		       (vcpu->arch.hfscr & HFSCR_TM)) {
		unsigned long msr = vcpu->arch.shregs.msr;
		if (MSR_TM_ACTIVE(msr)) {
			kvmppc_restore_tm_hv(vcpu, msr, true);
		} else {
			mtspr(SPRN_TEXASR, vcpu->arch.texasr);
			mtspr(SPRN_TFHAR, vcpu->arch.tfhar);
			mtspr(SPRN_TFIAR, vcpu->arch.tfiar);
		}
	}

	load_spr_state(vcpu, host_os_sprs);

	load_fp_state(&vcpu->arch.fp);
#ifdef CONFIG_ALTIVEC
	load_vr_state(&vcpu->arch.vr);
#endif
}
EXPORT_SYMBOL_GPL(load_vcpu_state);

void store_vcpu_state(struct kvm_vcpu *vcpu)
{
	store_spr_state(vcpu);

	store_fp_state(&vcpu->arch.fp);
#ifdef CONFIG_ALTIVEC
	store_vr_state(&vcpu->arch.vr);
#endif

	if ((cpu_has_feature(CPU_FTR_TM) ||
	    cpu_has_feature(CPU_FTR_P9_TM_HV_ASSIST)) &&
		       (vcpu->arch.hfscr & HFSCR_TM)) {
		unsigned long msr = vcpu->arch.shregs.msr;
		if (MSR_TM_ACTIVE(msr)) {
			kvmppc_save_tm_hv(vcpu, msr, true);
		} else {
			vcpu->arch.texasr = mfspr(SPRN_TEXASR);
			vcpu->arch.tfhar = mfspr(SPRN_TFHAR);
			vcpu->arch.tfiar = mfspr(SPRN_TFIAR);
		}
		vcpu->arch.hfscr &= ~HFSCR_TM;
	}
}
EXPORT_SYMBOL_GPL(store_vcpu_state);

void save_p9_host_os_sprs(struct p9_host_os_sprs *host_os_sprs)
{
	host_os_sprs->iamr = mfspr(SPRN_IAMR);
	host_os_sprs->amr = mfspr(SPRN_AMR);
}
EXPORT_SYMBOL_GPL(save_p9_host_os_sprs);

/* vcpu guest regs must already be saved */
void restore_p9_host_os_sprs(struct kvm_vcpu *vcpu,
				    struct p9_host_os_sprs *host_os_sprs)
{
	mtspr(SPRN_SPRG_VDSO_WRITE, local_paca->sprg_vdso);

	if (!cpu_has_feature(CPU_FTR_ARCH_31))
		mtspr(SPRN_TIDR, current->thread.tidr);
	if (host_os_sprs->iamr != vcpu->arch.iamr)
		mtspr(SPRN_IAMR, host_os_sprs->iamr);
	if (vcpu->arch.uamor != 0)
		mtspr(SPRN_UAMOR, 0);
	if (host_os_sprs->amr != vcpu->arch.amr)
		mtspr(SPRN_AMR, host_os_sprs->amr);
	if (current->thread.dscr != vcpu->arch.dscr)
		mtspr(SPRN_DSCR, current->thread.dscr);
	if (vcpu->arch.pspb != 0)
		mtspr(SPRN_PSPB, 0);

	/* Save guest CTRL register, set runlatch to 1 */
	if (!(vcpu->arch.ctrl & 1))
		mtspr(SPRN_CTRLT, 1);
}

static void switch_pmu_to_guest(struct kvm_vcpu *vcpu,
				    struct p9_host_os_sprs *host_os_sprs)
{
	if (ppc_get_pmu_inuse()) {
		bool do_isync = false;

		/*
		 * It might be better to put PMU handling (at least for the
		 * host) in the perf subsystem because it knows more about what
		 * is being used.
		 */

		/* POWER9, POWER10 do not implement HPMC or SPMC */

		host_os_sprs->mmcr0 = mfspr(SPRN_MMCR0);
		host_os_sprs->mmcra = mfspr(SPRN_MMCRA);

		if (host_os_sprs->mmcr0 != MMCR0_FC) {
			/* Freeze counters */
			mtspr(SPRN_MMCR0, MMCR0_FC);
			do_isync = true;
		}
		if (host_os_sprs->mmcra != 0) {
			/* Clear MMCRA in order to disable SDAR updates */
			mtspr(SPRN_MMCRA, 0);
			do_isync = true;
		}
		if (do_isync)
			isync();

		host_os_sprs->pmc1 = mfspr(SPRN_PMC1);
		host_os_sprs->pmc2 = mfspr(SPRN_PMC2);
		host_os_sprs->pmc3 = mfspr(SPRN_PMC3);
		host_os_sprs->pmc4 = mfspr(SPRN_PMC4);
		host_os_sprs->pmc5 = mfspr(SPRN_PMC5);
		host_os_sprs->pmc6 = mfspr(SPRN_PMC6);
		host_os_sprs->mmcr1 = mfspr(SPRN_MMCR1);
		host_os_sprs->mmcr2 = mfspr(SPRN_MMCR2);
		host_os_sprs->sdar = mfspr(SPRN_SDAR);
		host_os_sprs->siar = mfspr(SPRN_SIAR);
		host_os_sprs->sier1 = mfspr(SPRN_SIER);

		if (cpu_has_feature(CPU_FTR_ARCH_31)) {
			host_os_sprs->mmcr3 = mfspr(SPRN_MMCR3);
			host_os_sprs->sier2 = mfspr(SPRN_SIER2);
			host_os_sprs->sier3 = mfspr(SPRN_SIER3);
		}
	}

	if (vcpu->arch.hfscr & HFSCR_PM) {
		mtspr(SPRN_PMC1, vcpu->arch.pmc[0]);
		mtspr(SPRN_PMC2, vcpu->arch.pmc[1]);
		mtspr(SPRN_PMC3, vcpu->arch.pmc[2]);
		mtspr(SPRN_PMC4, vcpu->arch.pmc[3]);
		mtspr(SPRN_PMC5, vcpu->arch.pmc[4]);
		mtspr(SPRN_PMC6, vcpu->arch.pmc[5]);
		mtspr(SPRN_MMCR1, vcpu->arch.mmcr[1]);
		mtspr(SPRN_MMCR2, vcpu->arch.mmcr[2]);
		mtspr(SPRN_SDAR, vcpu->arch.sdar);
		mtspr(SPRN_SIAR, vcpu->arch.siar);
		mtspr(SPRN_SIER, vcpu->arch.sier[0]);

		if (cpu_has_feature(CPU_FTR_ARCH_31)) {
			mtspr(SPRN_MMCR3, vcpu->arch.mmcr[4]);
			mtspr(SPRN_SIER2, vcpu->arch.sier[1]);
			mtspr(SPRN_SIER3, vcpu->arch.sier[2]);
		}

		/* Set MMCRA then MMCR0 last */
		mtspr(SPRN_MMCRA, vcpu->arch.mmcra);
		mtspr(SPRN_MMCR0, vcpu->arch.mmcr[0]);
		/* No isync necessary because we're starting counters */
	}
}

static void switch_pmu_to_host(struct kvm_vcpu *vcpu,
				    struct p9_host_os_sprs *host_os_sprs)
{
	struct lppaca *lp;
	int save_pmu = 1;

	lp = vcpu->arch.vpa.pinned_addr;
	if (lp)
		save_pmu = lp->pmcregs_in_use;

	if (save_pmu) {
		bool do_isync = false;

		vcpu->arch.mmcr[0] = mfspr(SPRN_MMCR0);
		vcpu->arch.mmcra = mfspr(SPRN_MMCRA);

		if (vcpu->arch.mmcr[0] != MMCR0_FC) {
			/* Freeze counters */
			mtspr(SPRN_MMCR0, MMCR0_FC);
			do_isync = true;
		}
		if (vcpu->arch.mmcra != 0) {
			/* Clear MMCRA in order to disable SDAR updates */
			mtspr(SPRN_MMCRA, 0);
			do_isync = true;
		}
		if (do_isync)
			isync();

		vcpu->arch.pmc[0] = mfspr(SPRN_PMC1);
		vcpu->arch.pmc[1] = mfspr(SPRN_PMC2);
		vcpu->arch.pmc[2] = mfspr(SPRN_PMC3);
		vcpu->arch.pmc[3] = mfspr(SPRN_PMC4);
		vcpu->arch.pmc[4] = mfspr(SPRN_PMC5);
		vcpu->arch.pmc[5] = mfspr(SPRN_PMC6);
		vcpu->arch.mmcr[1] = mfspr(SPRN_MMCR1);
		vcpu->arch.mmcr[2] = mfspr(SPRN_MMCR2);
		vcpu->arch.sdar = mfspr(SPRN_SDAR);
		vcpu->arch.siar = mfspr(SPRN_SIAR);
		vcpu->arch.sier[0] = mfspr(SPRN_SIER);

		if (cpu_has_feature(CPU_FTR_ARCH_31)) {
			vcpu->arch.mmcr[3] = mfspr(SPRN_MMCR3);
			vcpu->arch.sier[1] = mfspr(SPRN_SIER2);
			vcpu->arch.sier[2] = mfspr(SPRN_SIER3);
		}

	} else if (vcpu->arch.hfscr & HFSCR_PM) {
		/*
		 * The guest accessed PMC SPRs without specifying they should
		 * be preserved. Stop them from counting if the guest had
		 * started anything.
		 */
		vcpu->arch.mmcr[0] = MMCR0_FC;
		vcpu->arch.mmcra = 0;
		if (mfspr(SPRN_MMCR0) != MMCR0_FC) {
			mtspr(SPRN_MMCR0, MMCR0_FC);
			isync();
		}
		if (mfspr(SPRN_MMCRA) != 0) {
			mtspr(SPRN_MMCRA, 0);
			isync();
		}

		/*
		 * Demand-fault PMU register access in the guest.
		 *
		 * This is used to grab the guest's VPA pmcregs_in_use value
		 * and reflect it into the host's VPA in the case of a nested
		 * hypervisor.
		 *
		 * It also avoids having to zero-out SPRs after each guest
		 * exit to avoid side-channels when.
		 *
		 * This is cleared here when we exit the guest, so later HFSCR
		 * interrupt handling can add it back to run the guest with
		 * PM enabled next time.
		 */
		vcpu->arch.hfscr &= ~HFSCR_PM;
	} else {
		/* XXX: fix for below warnings? */
		vcpu->arch.mmcr[0] = MMCR0_FC;
		vcpu->arch.mmcra = 0;
		/* Dev. paranoia. Should never happen. Remove me (or keep and fold into above case so it's fixed properly. */
#if 0
		these fire for some reason
		WARN_ON_ONCE(mfspr(SPRN_MMCR0) != MMCR0_FC);
		WARN_ON_ONCE(mfspr(SPRN_MMCRA) != 0);
#endif
	}

	if (ppc_get_pmu_inuse()) {
		mtspr(SPRN_PMC1, host_os_sprs->pmc1);
		mtspr(SPRN_PMC2, host_os_sprs->pmc2);
		mtspr(SPRN_PMC3, host_os_sprs->pmc3);
		mtspr(SPRN_PMC4, host_os_sprs->pmc4);
		mtspr(SPRN_PMC5, host_os_sprs->pmc5);
		mtspr(SPRN_PMC6, host_os_sprs->pmc6);
		mtspr(SPRN_MMCR1, host_os_sprs->mmcr1);
		mtspr(SPRN_MMCR2, host_os_sprs->mmcr2);
		mtspr(SPRN_SDAR, host_os_sprs->sdar);
		mtspr(SPRN_SIAR, host_os_sprs->siar);
		mtspr(SPRN_SIER, host_os_sprs->sier1);

		if (cpu_has_feature(CPU_FTR_ARCH_31)) {
			mtspr(SPRN_MMCR3, host_os_sprs->mmcr3);
			mtspr(SPRN_SIER2, host_os_sprs->sier2);
			mtspr(SPRN_SIER3, host_os_sprs->sier3);
		}

		/* Set MMCRA then MMCR0 last */
		mtspr(SPRN_MMCRA, host_os_sprs->mmcra);
		mtspr(SPRN_MMCR0, host_os_sprs->mmcr0);
		/* No need for isync because we're only starting counters */
	}
}

#ifdef CONFIG_KVM_BOOK3S_HV_EXIT_TIMING
static void __start_timing(struct kvm_vcpu *vcpu, struct kvmhv_tb_accumulator *next)
{
	struct kvmppc_vcore *vc = vcpu->arch.vcore;
	u64 tb = mftb() - vc->tb_offset_applied;

	vcpu->arch.cur_activity = next;
	vcpu->arch.cur_tb_start = tb;
}

static void __accumulate_time(struct kvm_vcpu *vcpu, struct kvmhv_tb_accumulator *next)
{
	struct kvmppc_vcore *vc = vcpu->arch.vcore;
	struct kvmhv_tb_accumulator *curr;
	u64 tb = mftb() - vc->tb_offset_applied;
	u64 prev_tb;
	u64 delta;
	u64 seq;

	curr = vcpu->arch.cur_activity;
	vcpu->arch.cur_activity = next;
	prev_tb = vcpu->arch.cur_tb_start;
	vcpu->arch.cur_tb_start = tb;

	if (!curr)
		return;

	delta = tb - prev_tb;

	seq = curr->seqcount;
	curr->seqcount = seq + 1;
	smp_wmb();
	curr->tb_total += delta;
	if (seq == 0 || delta < curr->tb_min)
		curr->tb_min = delta;
	if (delta > curr->tb_max)
		curr->tb_max = delta;
	smp_wmb();
	curr->seqcount = seq + 2;
}

#define start_timing(vcpu, next) __start_timing(vcpu, next)
#define end_timing(vcpu) __start_timing(vcpu, NULL)
#define accumulate_time(vcpu, next) __accumulate_time(vcpu, next)
#else
#define start_timing(vcpu, next) do {} while (0)
#define end_timing(vcpu) do {} while (0)
#define accumulate_time(vcpu, next) do {} while (0)
#endif

static inline void mfslb(unsigned int idx, u64 *slbee, u64 *slbev)
{
	asm volatile("slbmfev  %0,%1" : "=r" (*slbev) : "r" (idx));
	asm volatile("slbmfee  %0,%1" : "=r" (*slbee) : "r" (idx));
}

static inline void mtslb(u64 slbee, u64 slbev)
{
	asm volatile("slbmte %0,%1" :: "r" (slbev), "r" (slbee));
}

static inline void clear_slb_entry(unsigned int idx)
{
	mtslb(idx, 0);
}

static inline void slb_clear_invalidate_partition(void)
{
	clear_slb_entry(0);
	asm volatile(PPC_SLBIA(6));
}

/*
 * Malicious or buggy radix guests may have inserted SLB entries
 * (only 0..3 because radix always runs with UPRT=1), so these must
 * be cleared here to avoid side-channels. slbmte is used rather
 * than slbia, as it won't clear cached translations.
 */
static void radix_clear_slb(void)
{
	int i;

	for (i = 0; i < 4; i++)
		clear_slb_entry(i);
}

static void switch_mmu_to_guest_radix(struct kvm *kvm, struct kvm_vcpu *vcpu, u64 lpcr)
{
	struct kvm_nested_guest *nested = vcpu->arch.nested;
	u32 lpid;

	lpid = nested ? nested->shadow_lpid : kvm->arch.lpid;

	/*
	 * All the isync()s are overkill but trivially follow the ISA
	 * requirements. Some can likely be replaced with justification
	 * comment for why they are not needed.
	 */
	isync();
	mtspr(SPRN_LPID, lpid);
	isync();
	mtspr(SPRN_LPCR, lpcr);
	isync();
	mtspr(SPRN_PID, vcpu->arch.pid);
	isync();
}

static void switch_mmu_to_guest_hpt(struct kvm *kvm, struct kvm_vcpu *vcpu, u64 lpcr)
{
	u32 lpid;
	int i;

	lpid = kvm->arch.lpid;

	mtspr(SPRN_LPID, lpid);
	mtspr(SPRN_LPCR, lpcr);
	mtspr(SPRN_PID, vcpu->arch.pid);

	for (i = 0; i < vcpu->arch.slb_max; i++)
		mtslb(vcpu->arch.slb[i].orige, vcpu->arch.slb[i].origv);

	isync();
}

static void switch_mmu_to_host(struct kvm *kvm, u32 pid)
{
	isync();
	mtspr(SPRN_PID, pid);
	isync();
	mtspr(SPRN_LPID, kvm->arch.host_lpid);
	isync();
	mtspr(SPRN_LPCR, kvm->arch.host_lpcr);
	isync();

	if (!radix_enabled())
		slb_restore_bolted_realmode();
}

static void save_clear_host_mmu(struct kvm *kvm)
{
	if (!radix_enabled()) {
		/*
		 * Hash host could save and restore host SLB entries to
		 * reduce SLB fault overheads of VM exits, but for now the
		 * existing code clears all entries and restores just the
		 * bolted ones when switching back to host.
		 */
		slb_clear_invalidate_partition();
	}
}

static void save_clear_guest_mmu(struct kvm *kvm, struct kvm_vcpu *vcpu)
{
	if (kvm_is_radix(kvm)) {
		radix_clear_slb();
	} else {
		int i;
		int nr = 0;

		/*
		 * This must run before switching to host (radix host can't
		 * access all SLBs).
		 */
		for (i = 0; i < vcpu->arch.slb_nr; i++) {
			u64 slbee, slbev;
			mfslb(i, &slbee, &slbev);
			if (slbee & SLB_ESID_V) {
				vcpu->arch.slb[nr].orige = slbee | i;
				vcpu->arch.slb[nr].origv = slbev;
				nr++;
			}
		}
		vcpu->arch.slb_max = nr;
		slb_clear_invalidate_partition();
	}
}

int kvmhv_vcpu_entry_p9(struct kvm_vcpu *vcpu, u64 time_limit, unsigned long lpcr, u64 *tb)
{
	struct p9_host_os_sprs host_os_sprs;
	struct kvm *kvm = vcpu->kvm;
	struct kvm_nested_guest *nested = vcpu->arch.nested;
	struct kvmppc_vcore *vc = vcpu->arch.vcore;
	s64 hdec, dec;
	u64 purr, spurr;
	u64 *exsave;
	bool ri_set;
	int trap;
	unsigned long msr_facilities, msr;
	unsigned long host_hfscr;
	unsigned long host_ciabr;
	unsigned long host_dawr0;
	unsigned long host_dawrx0;
	unsigned long host_psscr;
	unsigned long host_pidr;
	unsigned long host_dawr1;
	unsigned long host_dawrx1;

	hdec = time_limit - *tb;
	if (hdec < 0)
		return BOOK3S_INTERRUPT_HV_DECREMENTER;

	WARN_ON_ONCE(vcpu->arch.shregs.msr & MSR_HV);
	WARN_ON_ONCE(!(vcpu->arch.shregs.msr & MSR_ME));

	start_timing(vcpu, &vcpu->arch.rm_entry);

	vcpu->arch.ceded = 0;

	host_hfscr = mfspr(SPRN_HFSCR);
	host_ciabr = mfspr(SPRN_CIABR);
	host_dawr0 = mfspr(SPRN_DAWR0);
	host_dawrx0 = mfspr(SPRN_DAWRX0);
	host_psscr = mfspr(SPRN_PSSCR);
	host_pidr = mfspr(SPRN_PID);
	if (cpu_has_feature(CPU_FTR_DAWR1)) {
		host_dawr1 = mfspr(SPRN_DAWR1);
		host_dawrx1 = mfspr(SPRN_DAWRX1);
	}

	local_paca->kvm_hstate.host_purr = mfspr(SPRN_PURR);
	local_paca->kvm_hstate.host_spurr = mfspr(SPRN_SPURR);

	save_p9_host_os_sprs(&host_os_sprs);

	/*
	 * This could be combined with MSR[RI] clearing, but that expands
	 * the unrecoverable window. It would be better to cover unrecoverable
	 * with KVM bad interrupt handling rather than use MSR[RI] at all.
	 *
	 * Much more difficult and less worthwhile to combine with IR/DR
	 * disable.
	 */
	hard_irq_disable();
	if (lazy_irq_pending()) {
		trap = 0;
		goto out;
	}

	/* MSR bits may have been cleared by context switch */
	msr_facilities = 0;
	if (IS_ENABLED(CONFIG_PPC_FPU))
		msr_facilities |= MSR_FP;
	if (cpu_has_feature(CPU_FTR_ALTIVEC))
		msr_facilities |= MSR_VEC;
	if (cpu_has_feature(CPU_FTR_VSX))
		msr_facilities |= MSR_VSX;
	if ((cpu_has_feature(CPU_FTR_TM) ||
	    cpu_has_feature(CPU_FTR_P9_TM_HV_ASSIST)) &&
			(vcpu->arch.hfscr & HFSCR_TM))
		msr_facilities |= MSR_TM;
	msr = msr_check_and_set(msr_facilities);
	/* Save MSR for restore. This is after hard disable, so EE is clear. */

	if (vc->tb_offset) {
		u64 new_tb = *tb + vc->tb_offset;
		mtspr(SPRN_TBU40, new_tb);
		if ((mftb() & 0xffffff) < (new_tb & 0xffffff)) {
			new_tb += 0x1000000;
			mtspr(SPRN_TBU40, new_tb);
		}
		*tb = new_tb;
		vc->tb_offset_applied = vc->tb_offset;
	}

	if (vc->pcr)
		mtspr(SPRN_PCR, vc->pcr | PCR_MASK);
	if (vc->dpdes)
		mtspr(SPRN_DPDES, vc->dpdes);

	if (dawr_enabled()) {
		if (vcpu->arch.dawr0 != host_dawr0)
			mtspr(SPRN_DAWR0, vcpu->arch.dawr0);
		if (vcpu->arch.dawrx0 != host_dawrx0)
			mtspr(SPRN_DAWRX0, vcpu->arch.dawrx0);
		if (cpu_has_feature(CPU_FTR_DAWR1)) {
			if (vcpu->arch.dawr1 != host_dawr1)
				mtspr(SPRN_DAWR1, vcpu->arch.dawr1);
			if (vcpu->arch.dawrx1 != host_dawrx1)
				mtspr(SPRN_DAWRX1, vcpu->arch.dawrx1);
		}
	}
	if (vcpu->arch.ciabr != host_ciabr)
		mtspr(SPRN_CIABR, vcpu->arch.ciabr);

	mtspr(SPRN_PSSCR, vcpu->arch.psscr | PSSCR_EC |
	      (local_paca->kvm_hstate.fake_suspend << PSSCR_FAKE_SUSPEND_LG));

	mtspr(SPRN_HFSCR, vcpu->arch.hfscr);

	mtspr(SPRN_HSRR0, vcpu->arch.regs.nip);
	mtspr(SPRN_HSRR1, (vcpu->arch.shregs.msr & ~MSR_HV) | MSR_ME);

	/*
	 * On POWER9 DD2.1 and below, sometimes on a Hypervisor Data Storage
	 * Interrupt (HDSI) the HDSISR is not be updated at all.
	 *
	 * To work around this we put a canary value into the HDSISR before
	 * returning to a guest and then check for this canary when we take a
	 * HDSI. If we find the canary on a HDSI, we know the hardware didn't
	 * update the HDSISR. In this case we return to the guest to retake the
	 * HDSI which should correctly update the HDSISR the second time HDSI
	 * entry.
	 *
	 * Just do this on all p9 processors for now.
	 */
//	mtspr(SPRN_HDSISR, HDSISR_CANARY);

	mtspr(SPRN_SPRG0, vcpu->arch.shregs.sprg0);
	mtspr(SPRN_SPRG1, vcpu->arch.shregs.sprg1);
	mtspr(SPRN_SPRG2, vcpu->arch.shregs.sprg2);
	mtspr(SPRN_SPRG3, vcpu->arch.shregs.sprg3);

	mtspr(SPRN_VTB, vc->vtb);
	mtspr(SPRN_PURR, vcpu->arch.purr);
	mtspr(SPRN_SPURR, vcpu->arch.spurr);
	mtspr(SPRN_IC, vcpu->arch.ic);

//	mtspr(SPRN_AMOR, ~0UL);

	local_paca->kvm_hstate.in_guest = KVM_GUEST_MODE_HV_P9;

	load_vcpu_state(vcpu, &host_os_sprs);

	/*
	 * Hash host, hash guest, or radix guest with prefetch bug, all have
	 * to disable the MMU before switching to guest MMU state.
	 */
	if (!radix_enabled() || !kvm_is_radix(kvm) ||
			cpu_has_feature(CPU_FTR_P9_RADIX_PREFETCH_BUG))
		__mtmsrd(msr & ~(MSR_IR|MSR_DR|MSR_RI), 0);

	save_clear_host_mmu(kvm);

	if (kvm_is_radix(kvm)) {
		switch_mmu_to_guest_radix(kvm, vcpu, lpcr);
		if (!cpu_has_feature(CPU_FTR_P9_RADIX_PREFETCH_BUG))
			__mtmsrd(0, 1); /* clear RI */

	} else {
		switch_mmu_to_guest_hpt(kvm, vcpu, lpcr);
	}

	/* TLBIEL uses LPID=LPIDR, so run this after setting guest LPID */
	kvmppc_check_need_tlb_flush(kvm, vc->pcpu, nested);

	/*
	 * P9 suppresses the HDEC exception when LPCR[HDICE] = 0,
	 * so set guest LPCR (with HDICE) before writing HDEC.
	 */
	mtspr(SPRN_HDEC, hdec);

	mtspr(SPRN_DEC, vcpu->arch.dec_expires - *tb);

	mtspr(SPRN_DAR, vcpu->arch.shregs.dar);
	mtspr(SPRN_DSISR, vcpu->arch.shregs.dsisr);
	mtspr(SPRN_SRR0, vcpu->arch.shregs.srr0);
	mtspr(SPRN_SRR1, vcpu->arch.shregs.srr1);

	accumulate_time(vcpu, &vcpu->arch.guest_time);

	switch_pmu_to_guest(vcpu, &host_os_sprs);
	kvmppc_p9_enter_guest(vcpu);
	switch_pmu_to_host(vcpu, &host_os_sprs);

	accumulate_time(vcpu, &vcpu->arch.rm_intr);

	/* XXX: Could get these from r11/12 and paca exsave instead */
	vcpu->arch.shregs.srr0 = mfspr(SPRN_SRR0);
	vcpu->arch.shregs.srr1 = mfspr(SPRN_SRR1);
	vcpu->arch.shregs.dar = mfspr(SPRN_DAR);
	vcpu->arch.shregs.dsisr = mfspr(SPRN_DSISR);

	/* 0x2 bit for HSRR is only used by PR and P7/8 HV paths, clear it */
	trap = local_paca->kvm_hstate.scratch0 & ~0x2;

	/* HSRR interrupts leave MSR[RI] unchanged, SRR interrupts clear it. */
	ri_set = false;
	if (likely(trap > BOOK3S_INTERRUPT_MACHINE_CHECK)) {
		if (trap != BOOK3S_INTERRUPT_SYSCALL &&
				(vcpu->arch.shregs.msr & MSR_RI))
			ri_set = true;
		exsave = local_paca->exgen;
	} else if (trap == BOOK3S_INTERRUPT_SYSTEM_RESET) {
		exsave = local_paca->exnmi;
	} else { /* trap == 0x200 */
		exsave = local_paca->exmc;
	}

	vcpu->arch.regs.gpr[1] = local_paca->kvm_hstate.scratch1;
	vcpu->arch.regs.gpr[3] = local_paca->kvm_hstate.scratch2;

	/*
	 * Only set RI after reading machine check regs (DAR, DSISR, SRR0/1)
	 * and hstate scratch (which we need to move into exsave to make
	 * re-entrant vs SRESET/MCE)
	 */
#if 0
	if (ri_set) {
		if (unlikely(!(mfmsr() & MSR_RI))) {
			__mtmsrd(MSR_RI, 1);
			WARN_ON_ONCE(1);
		}
	} else {
		WARN_ON_ONCE(mfmsr() & MSR_RI);
		__mtmsrd(MSR_RI, 1);
	}
#endif

	vcpu->arch.regs.gpr[9] = exsave[EX_R9/sizeof(u64)];
	vcpu->arch.regs.gpr[10] = exsave[EX_R10/sizeof(u64)];
	vcpu->arch.regs.gpr[11] = exsave[EX_R11/sizeof(u64)];
	vcpu->arch.regs.gpr[12] = exsave[EX_R12/sizeof(u64)];
	vcpu->arch.regs.gpr[13] = exsave[EX_R13/sizeof(u64)];
	vcpu->arch.ppr = exsave[EX_PPR/sizeof(u64)];
	vcpu->arch.cfar = exsave[EX_CFAR/sizeof(u64)];
	vcpu->arch.regs.ctr = exsave[EX_CTR/sizeof(u64)];

	vcpu->arch.last_inst = KVM_INST_FETCH_FAILED;

	if (unlikely(trap == BOOK3S_INTERRUPT_MACHINE_CHECK)) {
		vcpu->arch.fault_dar = exsave[EX_DAR/sizeof(u64)];
		vcpu->arch.fault_dsisr = exsave[EX_DSISR/sizeof(u64)];
		kvmppc_realmode_machine_check(vcpu);

	} else if (unlikely(trap == BOOK3S_INTERRUPT_HMI)) {
		kvmppc_realmode_hmi_handler();

	} else if (trap == BOOK3S_INTERRUPT_H_EMUL_ASSIST) {
		vcpu->arch.emul_inst = mfspr(SPRN_HEIR);

	} else if (trap == BOOK3S_INTERRUPT_H_DATA_STORAGE) {
		vcpu->arch.fault_dar = exsave[EX_DAR/sizeof(u64)];
		vcpu->arch.fault_dsisr = exsave[EX_DSISR/sizeof(u64)];
		vcpu->arch.fault_gpa = mfspr(SPRN_ASDR);

	} else if (trap == BOOK3S_INTERRUPT_H_INST_STORAGE) {
		vcpu->arch.fault_gpa = mfspr(SPRN_ASDR);

	} else if (trap == BOOK3S_INTERRUPT_H_FAC_UNAVAIL) {
		vcpu->arch.hfscr = mfspr(SPRN_HFSCR);

#ifdef CONFIG_PPC_TRANSACTIONAL_MEM
	/*
	 * Softpatch interrupt for transactional memory emulation cases
	 * on POWER9 DD2.2.  This is early in the guest exit path - we
	 * haven't saved registers or done a treclaim yet.
	 */
	} else if (trap == BOOK3S_INTERRUPT_HV_SOFTPATCH) {
		vcpu->arch.emul_inst = mfspr(SPRN_HEIR);

		/*
		 * The cases we want to handle here are those where the guest
		 * is in real suspend mode and is trying to transition to
		 * transactional mode.
		 */
		if (local_paca->kvm_hstate.fake_suspend &&
				(vcpu->arch.shregs.msr & MSR_TS_S)) {
			if (kvmhv_p9_tm_emulation_early(vcpu)) {
				/* Prevent it being handled again. */
				trap = 0;
			}
		}
#endif
	}

	accumulate_time(vcpu, &vcpu->arch.rm_exit);

	/* Advance host PURR/SPURR by the amount used by guest */
	purr = mfspr(SPRN_PURR);
	spurr = mfspr(SPRN_SPURR);
	vcpu->arch.purr = purr;
	vcpu->arch.spurr = spurr;

	vcpu->arch.ic = mfspr(SPRN_IC);
	vcpu->arch.pid = mfspr(SPRN_PID);
	vcpu->arch.psscr = mfspr(SPRN_PSSCR) & PSSCR_GUEST_VIS;

	vcpu->arch.shregs.sprg0 = mfspr(SPRN_SPRG0);
	vcpu->arch.shregs.sprg1 = mfspr(SPRN_SPRG1);
	vcpu->arch.shregs.sprg2 = mfspr(SPRN_SPRG2);
	vcpu->arch.shregs.sprg3 = mfspr(SPRN_SPRG3);

	vc->dpdes = mfspr(SPRN_DPDES);
	vc->vtb = mfspr(SPRN_VTB);

	dec = mfspr(SPRN_DEC);
	if (!(lpcr & LPCR_LD)) /* Sign extend if not using large decrementer */
		dec = (s32) dec;
	*tb = mftb();
	vcpu->arch.dec_expires = dec + *tb;

	if (vc->tb_offset_applied) {
		u64 new_tb = *tb - vc->tb_offset_applied;
		mtspr(SPRN_TBU40, new_tb);
		if ((mftb() & 0xffffff) < (new_tb & 0xffffff)) {
			new_tb += 0x1000000;
			mtspr(SPRN_TBU40, new_tb);
		}
		*tb = new_tb;
		vc->tb_offset_applied = 0;
	}

	save_clear_guest_mmu(kvm, vcpu);
	switch_mmu_to_host(kvm, host_pidr);

	/*
	 * Enable MSR here in order to have facilities enabled to save
	 * guest registers. This enables MMU (if we were in realmode), so
	 * only switch MMU on after the MMU is switched to host, to avoid
	 * the P9_RADIX_PREFETCH_BUG or hash guest context.
	 */
	__mtmsrd(msr, 0);

	store_vcpu_state(vcpu);

	mtspr(SPRN_PURR, local_paca->kvm_hstate.host_purr +
	      purr - vcpu->arch.purr);
	mtspr(SPRN_SPURR, local_paca->kvm_hstate.host_spurr +
	      spurr - vcpu->arch.spurr);

	/* Preserve PSSCR[FAKE_SUSPEND] until we've called kvmppc_save_tm_hv */
	mtspr(SPRN_PSSCR, host_psscr |
	      (local_paca->kvm_hstate.fake_suspend << PSSCR_FAKE_SUSPEND_LG));
	mtspr(SPRN_HFSCR, host_hfscr);
	if (vcpu->arch.ciabr != host_ciabr)
		mtspr(SPRN_CIABR, host_ciabr);
	if (vcpu->arch.dawr0 != host_dawr0)
		mtspr(SPRN_DAWR0, host_dawr0);
	if (vcpu->arch.dawrx0 != host_dawrx0)
		mtspr(SPRN_DAWRX0, host_dawrx0);
	if (cpu_has_feature(CPU_FTR_DAWR1)) {
		if (vcpu->arch.dawr1 != host_dawr1)
			mtspr(SPRN_DAWR1, host_dawr1);
		if (vcpu->arch.dawrx1 != host_dawrx1)
			mtspr(SPRN_DAWRX1, host_dawrx1);
	}

	if (vc->dpdes)
		mtspr(SPRN_DPDES, 0);
	if (vc->pcr)
		mtspr(SPRN_PCR, PCR_MASK);

	/* HDEC must be at least as large as DEC, so decrementer_max fits */
	mtspr(SPRN_HDEC, decrementer_max);

	timer_rearm_host_dec(*tb);

	restore_p9_host_os_sprs(vcpu, &host_os_sprs);

	local_paca->kvm_hstate.in_guest = KVM_GUEST_MODE_NONE;

	if (kvm_is_radix(kvm)) {
		/*
		 * Since this is radix, do a eieio; tlbsync; ptesync sequence
		 * in case we interrupted the guest between a tlbie and a
		 * ptesync.
		 */
		asm volatile("eieio; tlbsync; ptesync");
	}

	/*
	 * cp_abort is required if the processor supports local copy-paste
	 * to clear the copy buffer that was under control of the guest.
	 */
	if (cpu_has_feature(CPU_FTR_ARCH_31))
		asm volatile(PPC_CP_ABORT);

out:
	end_timing(vcpu);

	return trap;
}
EXPORT_SYMBOL_GPL(kvmhv_vcpu_entry_p9);
