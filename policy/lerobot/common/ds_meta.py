from numpy import array
from torch import tensor
from lerobot.configs.types import DictLike, FeatureType, PolicyFeature


# ds_meta_features = {
# 'wrist_image': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 648)), 'image': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)), 'state': PolicyFeature(type=FeatureType.STATE, shape=(8,)), 'actions': PolicyFeature(type=FeatureType.ACTION, shape=(8,)) }

ds_meta_features = { 'top_image': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 1080, 1920)), \
                    'wrist_image': PolicyFeature(FeatureType.VISUAL, shape=(3, 720, 1280)), \
                        'main_image': PolicyFeature(FeatureType.VISUAL, shape=(3, 720, 1280)), \
                            'human_image': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 1080, 1920)), \
                                'state': PolicyFeature(type=FeatureType.STATE, shape=(3,)), \
                                    'actions': PolicyFeature(type=FeatureType.ACTION, shape=(3,)), \
                                    'trajectory': PolicyFeature(type=FeatureType.TRAJ, shape=(3,)), \
                                        'state_trajectory': PolicyFeature(type=FeatureType.STATE, shape=(3,))}

# ds_meta_stats = {'actions': {
#     'mean': array([        -0.004510241565863962,
#         0.01614046322931098,
#         0.0014488363635850323,
#         0.0311631332509203,
#         0.001043915262106561,
#         -0.00020770560994670895,
#         0.004283504188299478,
#         0.4467688427843715]), 
#     'std': array([        0.15498387577776174,
#         0.30615082257104115,
#         0.15100500021006213,
#         0.30140113833297266,
#         0.22559903930268904,
#         0.24724309007821516,
#         0.26929824307690076,
#         0.44116568238342885])}, 
#     'state': { 'mean': array([        0.011335312069000092,
#         0.2720501795277465,
#         -0.01088267017501314,
#         -2.01687728054916,
#         -0.036302827384647,
#         2.3479571350249087,
#         0.09651094572890309,
#         0.39627546938562525]),
#           'std': array([0.31480194797319616,
#         0.4886056857285085,
#         0.2738964198316734,
#         0.48531355558421696,
#         0.5218106716806459,
#         0.4563026725101075,
#         0.7445646573567261,
#         0.40620681894754745])}}

ds_meta_stats = {'index': {'min': array([0]), 'max': array([36496]), 'mean': array([18248.]), 'std': array([10535.77638335]), 'count': array([36497])}, 'timestamp': {'min': array([0.]), 'max': array([8.5]), 'mean': array([2.7181266]), 'std': array([1.69467536]), 'count': array([36497])}, 'frame_index': {'min': array([0]), 'max': array([255]), 'mean': array([81.54379812]), 'std': array([50.84026083]), 'count': array([36497])}, 'wrist_image': {'min': array([[[0.]],

       [[0.]],

       [[0.]]]), 'max': array([[[1.]],

       [[1.]],

       [[1.]]]), 'mean': tensor([[[0.4850]],

        [[0.4560]],

        [[0.4060]]]), 'std': tensor([[[0.2290]],

        [[0.2240]],

        [[0.2250]]]), 'count': array([23052])}, 'state': {'min': array([ 0.19884486, -0.17603808, -0.19609748]), 'max': array([0.91320133, 0.41313821, 0.35258827]), 'mean': array([0.51127664, 0.15969605, 0.09030153]), 'std': array([0.13225974, 0.07531852, 0.06800139]), 'count': array([36497])}, \
            'actions': {'min': array([-0.10499817, -0.08844563, -0.2564878 ]), 'max': array([0.10414094, 0.08809033, 0.28673574]), 'mean': array([-0.00025619,  0.00042287,  0.00036644]), 'std': array([0.00580972, 0.00340703, 0.00947552]), 'count': array([36497])}, 'main_image': {'min': array([[[0.]],

       [[0.]],

       [[0.]]]), 'max': array([[[1.]],

       [[1.]],

       [[1.]]]), 'mean': tensor([[[0.4850]],

        [[0.4560]],

        [[0.4060]]]), 'std': tensor([[[0.2290]],

        [[0.2240]],

        [[0.2250]]]), 'count': array([23052])}, 'trajectory': {'min': array([ 0.19884486, -0.17603808, -0.19609748]), 'max': array([0.91320133, 0.41313821, 0.35258827]), 'mean': array([0.51127664, 0.15969605, 0.09030153]), 'std': array([0.13225974, 0.07531852, 0.06800139]), 'count': array([36497])}, 'episode_index': {'min': array([0]), 'max': array([230]), 'mean': array([112.54388032]), 'std': array([66.15593903]), 'count': array([36497])}, 'task_index': {'min': array([0]), 'max': array([1]), 'mean': array([0.39285421]), 'std': array([0.48838487]), 'count': array([36497])}, 'human_image': {'min': array([[[0.]],

       [[0.]],

       [[0.]]]), 'max': array([[[1.]],

       [[1.]],

       [[1.]]]), 'mean': tensor([[[0.4850]],

        [[0.4560]],

        [[0.4060]]]), 'std': tensor([[[0.2290]],

        [[0.2240]],

        [[0.2250]]]), 'count': array([23052])}, 'top_image': {'min': array([[[0.]],

       [[0.]],

       [[0.]]]), 'max': array([[[1.]],

       [[1.]],

       [[1.]]]), 'mean': tensor([[[0.4850]],

        [[0.4560]],

        [[0.4060]]]), 'std': tensor([[[0.2290]],

        [[0.2240]],

        [[0.2250]]]), 'count': array([23052])}, 'state_trajectory': {'min': array([ 0.19884486, -0.17603808, -0.19609748]), 'max': array([0.91320133, 0.41313821, 0.35258827]), 'mean': array([0.51127664, 0.15969605, 0.09030153]), 'std': array([0.13225974, 0.07531852, 0.06800139]), 'count': array([36497])}}

ds_meta_offline = {
    "features": ds_meta_features,
    "stats": ds_meta_stats
}