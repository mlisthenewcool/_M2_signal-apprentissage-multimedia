awk -F, '$8 ~ "img/"{print $8}' msdi_mapping.csv
