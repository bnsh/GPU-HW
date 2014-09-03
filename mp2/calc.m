function ssq = calc(dir)
	input0fn = strcat(dir,'/input0.raw');
	input1fn = strcat(dir,'/input1.raw');
	outputfn = strcat(dir,'/output.raw');
	id1 = tic();
	input0 = single(importdata(input0fn,' ',1).data);
	input1 = single(importdata(input1fn,' ',1).data);
	output = single(importdata(outputfn,' ',1).data);
	readfiles_time = toc(id1);
	fprintf(stderr, "Took %.7f seconds to read file\n", readfiles_time);
	id2 = tic();
	mul = (input0 * input1);
	multiply_time = toc(id2)
	fprintf(stderr, "Took %.7f seconds to multiply\n", multiply_time);
	err = mul - output;
	err2 = err .* err;
	ssq = sum(sum(err2));
endfunction
