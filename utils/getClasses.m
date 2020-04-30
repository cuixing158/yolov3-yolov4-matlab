function classes = getClasses(labelsPath)
fid = fopen(labelsPath,'r');
s = textscan(fid,'%s', 'Delimiter',{'    '});
s = s{1};
% classes =cellfun(@(x)x(11:end),s,'UniformOutput',false);
classes = s;
fclose(fid);