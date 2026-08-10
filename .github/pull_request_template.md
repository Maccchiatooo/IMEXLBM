<!--
Keep the title in the imperative and under ~50 characters.
Delete any section that does not apply.
-->

## What and why

<!-- The diff already says what changed. Say why it needed to change. -->

## Where it was tested

<!--
CI covers the CPU build only -- GitHub runners have no GPUs, so it cannot tell
you whether the Aurora, Polaris or Frontier build still links.

If this touches src/ or a machine Makefile, name the machine you built and ran
on, and paste the result. If you could not test on a machine, say so plainly so
a reviewer knows what is still unverified.
-->

- [ ] `./tests/regression.sh build/imexlbm` passes locally
- [ ] Built on: <!-- Aurora / Polaris / Frontier / CPU only -->

## Notes for the reviewer

<!-- Anything non-obvious: a tolerance you changed and why, a flag whose
     rationale is in a Makefile comment, a follow-up you deliberately left out. -->
