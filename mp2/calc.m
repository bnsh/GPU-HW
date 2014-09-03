function ssq = calc(dir)
	input0fn = strcat(dir,'/input0.raw');
	input1fn = strcat(dir,'/input1.raw');
	outputfn = strcat(dir,'/output.raw');
	input0 = single(importdata(input0fn,' ',1).data);
	input1 = single(importdata(input1fn,' ',1).data);
	output = single(importdata(outputfn,' ',1).data);
	tic
	mul = (input0 * input1);
	toc
	err = mul - output;
	err2 = err .* err;
	ssq = sum(sum(err2));
endfunction
