function create_DoG_MNIST(filename)

% read mnist
fp = fopen(filename, 'r+');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'uint8');
fclose(fp);

% DoG filter
images = reshape(images, numCols, numRows, numImages);

DoGfilter = DoG(7,1,2);
images_on = zeros(size(images));
images_off = zeros(size(images));

for i = 1:numImages
    image = imfilter(images(:,:,i), DoGfilter);
    
    image_on = image;
    image_on(image_on < 15) = 0;
    image_on = 255 * mat2gray(image_on);
    images_on(:,:,i) = image_on;
    
    image_off = -image;
    image_off(image_off < 20) = 0;
    image_off = 255 * mat2gray(image_off);
    images_off(:,:,i) = image_off;
end

% write to file
copyfile(filename, [filename '-DoG-ON']);
copyfile(filename, [filename '-DoG-OFF']);

fp = fopen([filename '-DoG-ON'], 'r+');
fseek(fp, 16, 'bof');
fwrite(fp, images_on, 'uint8');
fclose(fp);

fp = fopen([filename '-DoG-OFF'], 'r+');
fseek(fp, 16, 'bof');
fwrite(fp, images_off, 'uint8');
fclose(fp);

end
