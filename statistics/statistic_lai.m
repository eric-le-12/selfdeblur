
clear all

otherspath='../results/lai/others/';
gt_path='../groundtruth/';

fergus06=fullfile(otherspath,'06_fergus/');
cho09=fullfile(otherspath,'09_cho/');
xu10=fullfile(otherspath,'10_xu/');
krishnan11=fullfile(otherspath,'11_krishnan/');
levin11=fullfile(otherspath,'11_levin/');
sun13=fullfile(otherspath,'13_sun/');
xu13uniform=fullfile(otherspath,'13_xu_uniform/');
zhang13=fullfile(otherspath,'13_zhang/');
zhong13=fullfile(otherspath,'13_zhong/');
michaeli14=fullfile(otherspath,'14_michaeli/');
pan14=fullfile(otherspath,'14_pan/');
perrone14=fullfile(otherspath,'14_perrone/');
dark = fullfile(otherspath,'pan_DCP/');

selfdeblur='../result/';

a = pwd
struct_model = {
    struct('name','selfdeblur','path',selfdeblur),...
    };
nmodel = length(struct_model);

nimgs = 5;
nkernels=4;
maxshift=10;%Usually maxshift=5 is enough. If you find very low PSNR and SSIM for images with visually good results, maxshift should be set as a larger value. 

set = {'manmade','natural','people','saturated','text'};

for nnn = 1:nmodel
    modelpath = struct_model{nnn}.path;
    for mmm = 1:length(set)
        mname = set{mmm};
        for iii=1:nimgs
            for jjj=1:nkernels
                %         fprintf('img=%d,kernel=%d\n',iii,jjj);
                imgpath = fullfile(gt_path,sprintf('%s_%02d.png',mname,iii));
                x_true=im2double(imread(imgpath));%x_true
                if size(x_true,3) == 3
                    x_true = rgb2ycbcr(x_true);
                    x_true = x_true(:,:,1);
                end
                
                %our
                imgpath = fullfile(modelpath,sprintf('%s_%02d_kernel_%02d_x.png',mname,iii,jjj));
                deblur=im2double(imread(imgpath));%deblur
                orideblur = deblur;
                if size(deblur,3)==3
                    tmp = rgb2ycbcr(orideblur);
                    deblur = tmp(:,:,1);
                end
                
                [tp,ts,tI] = comp_upto_shift(deblur,x_true,maxshift); % tI is the aligned and cropped image. 

                
                imgpath = fullfile(modelpath,'aligned');
                if ~exist(imgpath,'dir')
                    mkdir(imgpath);
                end
                imgpath=fullfile(imgpath,sprintf('%s_%02d_kernel_%02d.png',mname,iii,jjj));
                if size(orideblur,3)==3
                    tmp = rgb2ycbcr(orideblur(16:end-15,16:end-15,:));
                    tmp(:,:,1) = tI;
                    tI = ycbcr2rgb(tmp);
                end
                imwrite(tI, imgpath)
                
                psnrs(mmm,iii,jjj)=tp;ssims(mmm,iii,jjj)=ts;
                
                fprintf('img=%d kernel=%d: psnr=%6.4f, ssim=%6.4f\n',iii,jjj,tp,ts);
                
            end
        end
    end
    
    fprintf('%s: psnr=%6.4f, ssim=%6.4f\n',struct_model{nnn}.name,mean(psnrs(:)),mean(ssims(:)));
    
    save(fullfile(struct_model{nnn}.path,'psnrs.mat'),'psnrs');
    save(fullfile(struct_model{nnn}.path,'ssims.mat'),'ssims');
       
end

