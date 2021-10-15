#!/system/bin/sh

#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: json_2_mask.sh
#
#   @Author: Shun Li
#
#   @Date: 2021-10-14
#
#   @Email: 2015097272@qq.com
#
#   @Description: 
#
#------------------------------------------------------------------------------

rm -rf label_images/

mkdir label_images/

cd json_images/
files=$(ls)

for file_name in $files
do
    echo "find $file_name"
    labelme_json_to_dataset $file_name

    cp -r ${file_name%%.*}"_json"/img.png ../images/${file_name%%.*}".png"
    cp -r ${file_name%%.*}"_json"/label.png ../label_images/${file_name%%.*}".png"

done
