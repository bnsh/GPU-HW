function ssq = calc(dir)
	input0fn = strcat(dir,'/input0.raw');
	input1fn = strcat(dir,'/input1.raw');
	outputfn = strcat(dir,'/output.raw');
	input0 = importdata(input0fn,' ',1).data;
	input1 = importdata(input1fn,' ',1).data;
	output = importdata(outputfn,' ',1).data;
	err = (input0 * input1) - output;
	err2 = err .* err;
	ssq = sum(sum(err2));
endfunction
