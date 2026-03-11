library(warbleR)
library(tuneR)
library(dplyr)
library(fs)

# source_dir <- "C:/Users/hfangyu/Desktop/OLD_segmented"
target_dir <- "D:\\Aging bird project\\1. Old-Young same individual\\159\\O_output_syllable_clips"
# dir_create(target_dir)
# 
# 
# syllable_files <- list.files(source_dir, pattern = "\\.wav$", recursive = TRUE, full.names = TRUE)
# 

# for (file in syllable_files) {
#   song_id <- basename(dirname(file))
#   syllable_name <- basename(file)
#   new_name <- paste0(song_id, "_", syllable_name)
#   file_copy(file, file.path(target_dir, new_name), overwrite = TRUE)
# }

# data
all_wavs <- dir_ls(target_dir, recurse = TRUE, type = "file", glob = "*.wav")
#create a flat dir
flat_dir <- file.path(target_dir, "ALL_WAVS_FLAT") 
#dir_create(flat_dir)
#file_copy(all_wavs, flat_dir, overwrite = TRUE)

setwd(flat_dir)
#syllable_files <- list.files(target_dir, pattern = "\\.wav$", full.names = TRUE)
X <- data.frame(
  sound.files = list.files(flat_dir, pattern = "\\.wav$", full.names = FALSE),
  selec = 1,
  start = 0,
  end = NA,
  stringsAsFactors = FALSE
)

# syllable time
'for (i in 1:nrow(X)) {
  wav <- readWave(file.path(target_dir, X$sound.files[i]))
  X$end[i] <- length(wav@left) / wav@samp.rate
}'

for (i in seq_len(nrow(X))) { 
  wav <- readWave(file.path(flat_dir, X$sound.files[i])) # 使用完整路径 
  X$end[i] <- length(wav@left) / wav@samp.rate 
}



# parameter
features <- spectro_analysis(
  X = X, 
  bp = c(0.25, 8), 
  threshold = 5, 
  wl = 512, 
  path = flat_dir
)

# save
write.csv(features, "C:\\Users\\lyuxuan\\Project_Code/159O.csv", row.names = FALSE)
print("Done")
