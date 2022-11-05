import torch

def gradientFromGray(img):
	
	kernel = torch.tensor(
		[\
		[ \
			[ \
				[0,0,0],\
				[0,1,-1],\
				[0,0,0]\
			] \
		], \
		[ \
			[ \
				[0,0,0],\
				[0,1,0],\
				[0,-1,0]\
			] \
		] \
	],dtype=torch.float32
	).cuda() # 2, 1, 3, 3

	return torch.nn.functional.conv2d(input=img, weight=kernel, stride=1, padding="same")
