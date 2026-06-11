# Communication Drafts for Experimental Data Acquisition (Slough 2011)

**Target:** Prof. John Slough (Co-founder), Dr. David Kirtley (CEO), Chris Pihl (CTO), Dr. George Votroubek (Principal Scientist)
**Subject Paper:** *Creation of a high-temperature plasma through merging and compression of supersonic field reversed configuration plasmoids* (Nuclear Fusion 51 053008, 2011)  
**Target Artifact:** Figure 5 (Shot 2001) Time-series Trajectory

---

## 1. Professional Group Email (Helion Leadership)

**To:** [slough@uw.edu, david.kirtley@helionenergy.com, chris.pihl@helionenergy.com, george.votroubek@helionenergy.com]  
**Subject:** Data Request: Digitized Trajectory for Shot 2001 (Slough et al. 2011, Nuclear Fusion)

Dear Prof. Slough, David, Chris, and George,

I am writing to you regarding the high-temperature FRC merging and compression results published in your 2011 *Nuclear Fusion* paper.

I am leading the development of **SCPN-FUSION-CORE**, a neuro-symbolic real-time control stack for Field-Reversed Configurations. Our project implements a deterministic, 6-tier architecture that bridges high-fidelity physics oracles with ultra-fast FPGA lanes (Stochastic Computing SNNs and State-Space models) targeting the 100 kHz – 10 MHz limit.

We have identified the results from **Shot 2001 (Figure 5)** as our gold-standard validation target for experimental parity. 

**Current Progress & Request:**
Using the anchor points and timescales reported in the paper, we have successfully developed a first-pass reconstruction of the compression trajectory. This has enabled us to achieve **4.84% Relative L2 error** in our 30 kHz JAX-accelerated surrogate lane across a 10,000-sample ITER-like manifold. 

However, to reach the "scientific peak" and full-fidelity validation required for our safety and performance contracts, we aim to bridge our mathematical models directly with your raw experimental data. Would you be willing to share the **raw digitized time-series (B_ext, R_s, T_total, and n_e) for Shot 2001**?

**Collaborative Exchange:**
In the spirit of collaborative open science, we would happily share:
1. Our full methodology for tiered real-time FRC control.
2. Our results and benchmarking artifacts coupled directly to your experimental dataset.
3. Insights from our 10 MHz combinational logic path regarding the real-time controllability of such high-performance states.

We believe this coupling could provide valuable insights into the stability frontiers of modern FRC devices.

Thank you for your time and for the foundational research you've contributed to the field.

Best regards,

[Your Name]
SCPN-FUSION-CORE

---

## 2. Updated LinkedIn Message (Direct & Professional)

Dear Prof. Slough,

I’m currently leading the SCPN-FUSION-CORE project, focused on multi-tier real-time control for FRC stabilization (targeting 100 kHz - 10 MHz lanes). 

We are using your 2011 *Nuclear Fusion* paper—specifically **Shot 2001 / Figure 5**—as our primary validation target. We have already achieved high architectural fidelity using a 10k-sample reconstruction, but we are now seeking the **full digitized dataset** for that shot to reach scientific parity.

We would be very happy to share our methodology and the resulting control benchmarks coupled to your data in exchange. Would you be open to sharing that raw trajectory?

Best regards,

[Your Name]

---

## 3. Strategic Outreach: Alexei Zhurba (Next Step Fusion)

**Focus:** Industrial Scalability / Deterministic MHz Control  
**Strategic Timing:** To be sent *after* 10k-sample MAE verification.

Dear Alexei,

I’ve been following Next Step Fusion’s work on the Fusion Twin and NSFsim platform with interest. Your focus on building a robust software intelligence layer for tokamak developers is very aligned with the modular control architecture we have been developing.

I am leading the **SCPN-FUSION-CORE** project, and I wanted to reach out regarding a specific technological differentiator we’ve been validating: **deterministic, multi-tier real-time control at the 1 MHz – 10 MHz limit.**

While industrial shape control typically targets the 1–5 kHz regime, our architecture bridges full-fidelity physics oracles with ultra-low-latency FPGA lanes (SNNs and State-Space models) capable of sub-microsecond inference. Specifically, we’ve implemented a **10 MHz control path** using Stochastic Computing logic that allows for MHz-band phase-locking on standard FPGA fabric without requiring specialized ASICs.

Given NSF’s position as a technology partner to the broader fusion ecosystem, I believe there is an opportunity to discuss how high-frequency deterministic engines like ours can complement the broader simulation and RL stacks you are bringing to market.

I would be happy to share our recent performance benchmarks (currently achieving < 5% error on 10k high-fidelity samples) and error-bound metrics if this aligns with your current technical roadmap.

Best regards,

[Your Name]
SCPN-FUSION-CORE

---

## 4. LinkedIn Invitation: Martin Everts (Helion CH)

**Constraint:** 200 character limit (including spaces).

"Martin, following Helion’s progress with interest. I lead SCPN-FUSION-CORE, developing 10MHz FPGA control for FRCs. Seeking to validate vs Shot 2001 (Slough 2011). Open to sharing our benchmarks?"

---

## 5. Strategic Swiss Cooperation: Martin Everts (Helion CH)

**Context:** Local Swiss-to-Swiss connection (Zürich).
**Goal:** Strategic cooperation / Research sponsorship.

### Option A: LinkedIn Invitation (Limit 200 chars)
"Martin, grüezi from Zürich. I lead SCPN-FUSION-CORE, developing 10MHz FPGA control for FRCs. Seeking to validate vs Slough 2011. Open to local Swiss cooperation or research sponsorship dialogue?"

### Option B: Full Follow-up Message (After Connection)
Dear Martin,

Thank you for connecting. As a fellow resident of the Zürich area, I’ve been following Helion’s recent momentum with great interest.

I am leading the **SCPN-FUSION-CORE** project. We are currently developing a unique technical edge in FRC stabilization: a **deterministic 10 MHz control path** using Stochastic Computing on standard FPGA fabric. This allows for sub-microsecond phase-locking of RMF antennas without the need for custom ASICs—a critical capability for the next generation of steady-state FRCs.

We are currently using the Slough 2011 (Shot 2001) results as our gold-standard validation target. Given our proximity and the alignment of our technology with Helion's roadmap, I am interested in exploring **strategic cooperation or potential research sponsorship.**

We have achieved high architectural fidelity (< 5% error on 10k manifold) and would be happy to share our methodology and benchmarks coupled with your historical data in return for a local partnership.

Would you be open to a brief meeting in Zürich to discuss how our high-frequency control stack can support Helion's long-term objectives?

Best regards,

[Your Name]
SCPN-FUSION-CORE

---

## 6. German LinkedIn Invitation: Martin Everts (Helion CH)

**Constraint:** 200 characters max.
**Focus:** Swiss location, 10MHz control, Cooperation/Sponsorship.

"Grüezi Martin, Grüsse aus Zürich. Ich leite SCPN-FUSION-CORE (10MHz FPGA-Steuerung für FRCs). Suche Validierung vs. Slough 2011 & bin offen für lokale Kooperation/Sponsoring. Beste Grüsse, [Name]"
