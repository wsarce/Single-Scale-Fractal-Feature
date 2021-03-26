% https://stackoverflow.com/a/2654459
dirName = 'C:\Users\Walker Arce\Documents\Business\Research\Papers\Fractals\A Method for Calculating the Fractal Dimension of Pixel Based Images\Final Paper\Data\Software\VRSTR';
filelist = dir(fullfile(dirName, '**\*.*'));  %get list of files and folders in any subfolder
filelist = filelist(~[filelist.isdir]);  %remove folders from list

for k = 1 : length(filelist)
    filelist(k).name
    filepath = fullfile(filelist(k).folder, filelist(k).name);
    probboxcountingcolorw3(filepath)
end