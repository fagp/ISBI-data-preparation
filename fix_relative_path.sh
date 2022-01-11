current_dir=$(pwd)

sed 's,\.\.,'"$current_dir"',g' ${current_dir}/train_scripts/configuration/dataconfig_train_backup.json > ${current_dir}/train_scripts/configuration/dataconfig_train.json
