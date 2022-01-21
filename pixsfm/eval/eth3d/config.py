from pathlib import Path

DATASET_PATH = Path("./datasets/ETH3D/")
OUTPUTS_PATH = Path("./outputs/ETH3D/")


OUTDOOR = [
    "courtyard",
    "electro",
    "facade",
    "meadow",
    "playground",
    "terrace"
]
INDOOR = [
    "delivery_area",
    "kicker",
    "office",
    "pipes",
    "relief",
    "relief_2",
    "terrains",
]
SCENES = OUTDOOR + INDOOR

preprocessing = {
    'resize_max': 1600,
    'interpolation': 'cv2_area',  # TODO: change to pil_linear
}
feature_configs = {
    "sift": {
        'model': {
            'name': 'sift',
            'num_octaves': 4,
            'octave_resolution': 3,
            'first_octave': 0,
            'edge_thresh': 10,
            'peak_thresh': 0.0066666666666666671,  # from COLMAP
            'upright': False,
            'root': True,
            'max_keypoints': -1
        },
        'preprocessing': {
            'grayscale': True,
            **preprocessing,
        },
    },
    "superpoint": {
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': -1,
            'keypoint_threshold': 0.015,
        },
        'preprocessing': {
            'grayscale': True,
            **preprocessing,
        },
    },
    "r2d2": {
        'model': {
            'name': 'r2d2',
            'model_name': 'r2d2_WAF_N16.pt',
            'max_keypoints': 5000,
            'scale_factor': 2**0.25,
            'min_size': 256,
            'max_size': 1600,
            'min_scale': 0,
            'max_scale': 1,
            'reliability_threshold': 0.7,
            'repetability_threshold': 0.7,
        },
        'preprocessing': {
            'grayscale': False,
            **preprocessing,
        },
    },
    "d2-net": {
        'model': {
            'name': 'd2net',
            'multiscale': False,
        },
        'preprocessing': {
            'grayscale': False,
            **preprocessing,
        },
    },
}

match_configs = {
    "sift": {
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'ratio_threshold': 0.8,
        }
    },
    "superpoint": {
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': (2 * (1-0.755))**0.5,  # from similarity
        }
    },
    "d2-net": {
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': (2 * (1-0.8))**0.5,  # from similarity
        }
    },
    "r2d2": {
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': (2 * (1-0.9))**0.5,  # from similarity
        }
    },
}

FEATURES = list(feature_configs.keys())
DEFAULT_FEATURES = ["sift", "superpoint", "r2d2"]

LOCALIZATION_IMAGES = {
   "pipes": [
      "dslr_images_undistorted/DSC_0643.JPG",
      "dslr_images_undistorted/DSC_0645.JPG",
      "dslr_images_undistorted/DSC_0647.JPG",
      "dslr_images_undistorted/DSC_0640.JPG",
      "dslr_images_undistorted/DSC_0636.JPG",
      "dslr_images_undistorted/DSC_0638.JPG",
      "dslr_images_undistorted/DSC_0642.JPG",
      "dslr_images_undistorted/DSC_0635.JPG",
      "dslr_images_undistorted/DSC_0644.JPG",
      "dslr_images_undistorted/DSC_0641.JPG"
   ],
   "courtyard": [
      "dslr_images_undistorted/DSC_0304.JPG",
      "dslr_images_undistorted/DSC_0287.JPG",
      "dslr_images_undistorted/DSC_0298.JPG",
      "dslr_images_undistorted/DSC_0308.JPG",
      "dslr_images_undistorted/DSC_0312.JPG",
      "dslr_images_undistorted/DSC_0302.JPG",
      "dslr_images_undistorted/DSC_0297.JPG",
      "dslr_images_undistorted/DSC_0313.JPG",
      "dslr_images_undistorted/DSC_0307.JPG",
      "dslr_images_undistorted/DSC_0321.JPG"
   ],
   "playground": [
      "dslr_images_undistorted/DSC_0585.JPG",
      "dslr_images_undistorted/DSC_0568.JPG",
      "dslr_images_undistorted/DSC_0579.JPG",
      "dslr_images_undistorted/DSC_0589.JPG",
      "dslr_images_undistorted/DSC_0593.JPG",
      "dslr_images_undistorted/DSC_0583.JPG",
      "dslr_images_undistorted/DSC_0578.JPG",
      "dslr_images_undistorted/DSC_0594.JPG",
      "dslr_images_undistorted/DSC_0588.JPG",
      "dslr_images_undistorted/DSC_0604.JPG"
   ],
   "delivery_area": [
      "dslr_images_undistorted/DSC_0717.JPG",
      "dslr_images_undistorted/DSC_0703.JPG",
      "dslr_images_undistorted/DSC_0714.JPG",
      "dslr_images_undistorted/DSC_0692.JPG",
      "dslr_images_undistorted/DSC_0686.JPG",
      "dslr_images_undistorted/DSC_0712.JPG",
      "dslr_images_undistorted/DSC_0715.JPG",
      "dslr_images_undistorted/DSC_0685.JPG",
      "dslr_images_undistorted/DSC_0702.JPG",
      "dslr_images_undistorted/DSC_0718.JPG"
   ],
   "terrace": [
      "dslr_images_undistorted/DSC_0271.JPG",
      "dslr_images_undistorted/DSC_0284.JPG",
      "dslr_images_undistorted/DSC_0268.JPG",
      "dslr_images_undistorted/DSC_0260.JPG",
      "dslr_images_undistorted/DSC_0267.JPG",
      "dslr_images_undistorted/DSC_0272.JPG",
      "dslr_images_undistorted/DSC_0259.JPG",
      "dslr_images_undistorted/DSC_0269.JPG",
      "dslr_images_undistorted/DSC_0262.JPG",
      "dslr_images_undistorted/DSC_0279.JPG"
   ],
   "meadow": [
      "dslr_images_undistorted/DSC_6559.JPG",
      "dslr_images_undistorted/DSC_6548.JPG",
      "dslr_images_undistorted/DSC_6541.JPG",
      "dslr_images_undistorted/DSC_6540.JPG",
      "dslr_images_undistorted/DSC_6535.JPG",
      "dslr_images_undistorted/DSC_6556.JPG",
      "dslr_images_undistorted/DSC_6558.JPG",
      "dslr_images_undistorted/DSC_6536.JPG",
      "dslr_images_undistorted/DSC_6539.JPG",
      "dslr_images_undistorted/DSC_6547.JPG"
   ],
   "electro": [
      "dslr_images_undistorted/DSC_9301.JPG",
      "dslr_images_undistorted/DSC_9289.JPG",
      "dslr_images_undistorted/DSC_9298.JPG",
      "dslr_images_undistorted/DSC_9274.JPG",
      "dslr_images_undistorted/DSC_9268.JPG",
      "dslr_images_undistorted/DSC_9296.JPG",
      "dslr_images_undistorted/DSC_9299.JPG",
      "dslr_images_undistorted/DSC_9267.JPG",
      "dslr_images_undistorted/DSC_9287.JPG",
      "dslr_images_undistorted/DSC_9302.JPG"
   ],
   "kicker": [
      "dslr_images_undistorted/DSC_6518.JPG",
      "dslr_images_undistorted/DSC_6496.JPG",
      "dslr_images_undistorted/DSC_6506.JPG",
      "dslr_images_undistorted/DSC_6503.JPG",
      "dslr_images_undistorted/DSC_6494.JPG",
      "dslr_images_undistorted/DSC_6492.JPG",
      "dslr_images_undistorted/DSC_6489.JPG",
      "dslr_images_undistorted/DSC_6504.JPG",
      "dslr_images_undistorted/DSC_6510.JPG",
      "dslr_images_undistorted/DSC_6490.JPG"
   ],
   "facade": [
      "dslr_images_undistorted/DSC_0347.JPG",
      "dslr_images_undistorted/DSC_0330.JPG",
      "dslr_images_undistorted/DSC_0396.JPG",
      "dslr_images_undistorted/DSC_0345.JPG",
      "dslr_images_undistorted/DSC_0390.JPG",
      "dslr_images_undistorted/DSC_0392.JPG",
      "dslr_images_undistorted/DSC_0341.JPG",
      "dslr_images_undistorted/DSC_0333.JPG",
      "dslr_images_undistorted/DSC_0352.JPG",
      "dslr_images_undistorted/DSC_0412.JPG"
   ],
   "office": [
      "dslr_images_undistorted/DSC_0253.JPG",
      "dslr_images_undistorted/DSC_0223.JPG",
      "dslr_images_undistorted/DSC_0237.JPG",
      "dslr_images_undistorted/DSC_0220.JPG",
      "dslr_images_undistorted/DSC_0239.JPG",
      "dslr_images_undistorted/DSC_0249.JPG",
      "dslr_images_undistorted/DSC_0229.JPG",
      "dslr_images_undistorted/DSC_0251.JPG",
      "dslr_images_undistorted/DSC_0221.JPG",
      "dslr_images_undistorted/DSC_0222.JPG"
   ],
   "relief": [
      "dslr_images_undistorted/DSC_0455.JPG",
      "dslr_images_undistorted/DSC_0435.JPG",
      "dslr_images_undistorted/DSC_0443.JPG",
      "dslr_images_undistorted/DSC_0440.JPG",
      "dslr_images_undistorted/DSC_0433.JPG",
      "dslr_images_undistorted/DSC_0431.JPG",
      "dslr_images_undistorted/DSC_0428.JPG",
      "dslr_images_undistorted/DSC_0441.JPG",
      "dslr_images_undistorted/DSC_0447.JPG",
      "dslr_images_undistorted/DSC_0429.JPG"
   ],
   "relief_2": [
      "dslr_images_undistorted/DSC_0487.JPG",
      "dslr_images_undistorted/DSC_0466.JPG",
      "dslr_images_undistorted/DSC_0474.JPG",
      "dslr_images_undistorted/DSC_0471.JPG",
      "dslr_images_undistorted/DSC_0464.JPG",
      "dslr_images_undistorted/DSC_0462.JPG",
      "dslr_images_undistorted/DSC_0459.JPG",
      "dslr_images_undistorted/DSC_0472.JPG",
      "dslr_images_undistorted/DSC_0478.JPG",
      "dslr_images_undistorted/DSC_0460.JPG"
   ],
   "terrains": [
      "dslr_images_undistorted/DSC_0626.JPG",
      "dslr_images_undistorted/DSC_0618.JPG",
      "dslr_images_undistorted/DSC_0649.JPG",
      "dslr_images_undistorted/DSC_0667.JPG",
      "dslr_images_undistorted/DSC_0661.JPG",
      "dslr_images_undistorted/DSC_0632.JPG",
      "dslr_images_undistorted/DSC_0650.JPG",
      "dslr_images_undistorted/DSC_0660.JPG",
      "dslr_images_undistorted/DSC_0619.JPG",
      "dslr_images_undistorted/DSC_0625.JPG"
   ]
}
