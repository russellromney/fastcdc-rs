# LLVM-MCA-BEGIN withchecks
.Lloop1:
	cmpq	%r8, %rdx
	jae	.Lexit1
	shlq	$2, %rax
	movzbl	(%rdi,%rdx), %esi
	addq	(%r10,%rsi,8), %rax
	testq	%r14, %rax
	je	.Lexit1
	leaq	1(%rdx), %rsi
	cmpq	%r8, %rsi
	jae	.Lexit1
	movzbl	1(%rdi,%rdx), %r15d
	addq	(%rcx,%r15,8), %rax
	testq	%r9, %rax
	je	.Lexit1
	addq	$2, %rdx
	decq	%rbx
	jne	.Lloop1
.Lexit1:
# LLVM-MCA-END
