python segmentation.py   --wav_dir "/home/yuxuan.li/workspace/Birdsong-project/Wav/finch/D17.853_noise/D17.853_O_songs"   --out_dir "/home/yuxuan.li/workspace/Birdsong-project/Wav/finch/D17.853_noise/D17.853_seg"

python MFCC.py   --segments_dir "/home/yuxuan.li/workspace/Birdsong-project/Wav/finch/D17.853_noise/D17.853_O_seg"   --wav_dir "/home/yuxuan.li/workspace/Birdsong-project/Wav/finch/D17.853_noise/D17.853_O_songs"   --out_dir "/home/yuxuan.li/workspace/Birdsong-project/Wav/finch/D17.853_noise/D17.853_O_MFCC"   --strip_prefix "a b c d e f i " --overwrite_segments_csv

#python name.py "/home/yuxuan.li/workspace/Birdsong-project/Wav/finch/train" "/home/yuxuan.li/workspace/Birdsong-project/Wav/finch/246_O_anno_y"

python annotation.py   --annotated_dir "/home/yuxuan.li/workspace/Birdsong-project/Wav/finch/D17.853_noise/D17.853_O_annotated_excel"   --segments_dir "/home/yuxuan.li/workspace/Birdsong-project/Wav/finch/D17.853_noise/D17.853_O_seg"   --output_txt_dir "/home/yuxuan.li/workspace/Birdsong-project/Wav/finch/D17.853_noise/D17.853_O_anno_y" --stats_csv /home/yuxuan.li/workspace/Birdsong-project/Wav/finch/D17.853_noise/D17.853_O_anno_y/stats.csv

python3 classification_compare.py   --features_dir "/home/yuxuan.li/workspace/Birdsong-project/Wav/finch/D17.853_noise/D17.853_O_MFCC"   --labels_dir "/home/yuxuan.li/workspace/Birdsong-project/Wav/finch/D17.853_noise/D17.853_O_anno_y"   --test_size 0.2   --random_state 42 --cv_folds 5 --save_model "/home/yuxuan.li/workspace/Birdsong-project/Wav/finch/246_O/best_model.joblib"
