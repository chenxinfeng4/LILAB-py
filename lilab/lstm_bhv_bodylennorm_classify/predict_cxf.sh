# conda activate mmdet
# 项目文件夹。包含 *400p.mp4, *.smoothed_foot.matcalibpkl
PROEJECT_DIR=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/24FP-behCluster/2407FP-total

# 动物在每个video里面算体长，算sniffzone
# output = 'bodylength.pkl'
python -m lilab.lstm_bhv_bodylennorm_classify.s0_prepare_body_length $PROEJECT_DIR    
python -m lilab.lstm_bhv_bodylennorm_classify.s01_matcalibpkl2rawfeatpkl $PROEJECT_DIR

# 按速度归一化
python -m lilab.lstm_bhv_bodylennorm_classify.s02_rawfeatpkl_to_norm $PROEJECT_DIR


# 从3D关键点出发，加上体长信息，算出行为分类
# output = 'lstm_offline.clippredpkl'
python -m lilab.lstm_bhv_bodylennorm_classify.s1_matcalibpkl2clippredpkl $PROEJECT_DIR


# ====和之前的流程一样
# 只看镜像一致性的视频片段，representative
# output = 'representitive_k36_filt_perc*/Representive_K36.clippredpkl'
python -m lilab.OpenLabCluster_train.a1_mirror_mutual_filt_clippredpkl $PROEJECT_DIR/lstm_offline.clippredpkl --already-mirrored


# 画出每个类别的视频片段
# output = 'representitive_k36_filt_perc*/400p_clusters/*.mp4'
python -m lilab.OpenLabCluster_train.a6b_clippredpkl_2_cluster400p  \
    $PROEJECT_DIR/representitive_k36_filt_perc*/Representive_K36.clippredpkl \
    $PROEJECT_DIR

# 根据 clippredpkl 生成 seqpkl
# output = 'representitive_k36_filt_perc*/400p_clusters/lstm_offline_sequences.pkl'
python -m lilab.OpenLabCluster_train.a6_clippredpkl_2_seqencepkl $PROEJECT_DIR/lstm_offline.clippredpkl --autoEnd
