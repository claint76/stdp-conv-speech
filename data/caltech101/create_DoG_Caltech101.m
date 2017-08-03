DoGfilter = DoG(7,1,2);

mkdir('101_ObjectCategories_resized_DoG');

for c = {'Faces', 'Motorbikes'}
    c = char(c);
    mkdir(fullfile('101_ObjectCategories_resized_DoG', c));
    
    files=dir(fullfile('101_ObjectCategories_resized', c, '*.jpg'));
    for k=1:length(files)
        disp(fullfile('101_ObjectCategories_resized', c, files(k).name));
        im = imread(fullfile('101_ObjectCategories_resized', c, files(k).name));
        
        if size(im, 3) == 3
            im = rgb2gray(im);
        end
        im = imfilter(im, DoGfilter);
        im(im < 15) = 0;
        im = 255 * mat2gray(im);
        
        imwrite(im, fullfile('101_ObjectCategories_resized_DoG', c, files(k).name));
    end
end