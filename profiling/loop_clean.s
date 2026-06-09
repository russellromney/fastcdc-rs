# LLVM-MCA-BEGIN clean
.Lloop2:
	shlq	$2, %rax
	movzbl	(%rdi,%rdx), %esi
	addq	(%r10,%rsi,8), %rax
	testq	%r14, %rax
	je	.Lexit2
	movzbl	1(%rdi,%rdx), %r15d
	addq	(%rcx,%r15,8), %rax
	testq	%r9, %rax
	je	.Lexit2
	addq	$2, %rdx
	decq	%rbx
	jne	.Lloop2
.Lexit2:
# LLVM-MCA-END
