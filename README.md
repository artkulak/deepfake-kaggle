# Solution for the DeepFake Competition Kaggle
https://www.kaggle.com/c/deepfake-detection-challenge

**Team: [ods.ai] LIFE IS FAKE**

**Solution got us to top 8% in the public LB**


* KUDOS to my teammates:
	- https://www.kaggle.com/blackitten13
	- https://www.kaggle.com/dmitriyab
  - https://www.kaggle.com/arli2016

## Model description
Pipeline: FaceDetection using BlazeFace -> Face classification with Efficient Net (Binary classification on Fake and Non Fake faces)

Our final solution is a blend of several Efficient Net architectures. We tried adding sound to our models as well, but it did not result in any improvement on the LB.

## Task description
This competition is closed for submissions. Participants' selected code submissions are being re-run by the host on a privately-held test set. Private leaderboard results are expected to be available as defined on the Timeline page.

Deepfake techniques, which present realistic AI-generated videos of people doing and saying fictional things, have the potential to have a significant impact on how people determine the legitimacy of information presented online. These content generation and modification technologies may affect the quality of public discourse and the safeguarding of human rights—especially given that deepfakes may be used maliciously as a source of misinformation, manipulation, harassment, and persuasion. Identifying manipulated media is a technically demanding and rapidly evolving challenge that requires collaborations across the entire tech industry and beyond.


AWS, Facebook, Microsoft, the Partnership on AI’s Media Integrity Steering Committee, and academics have come together to build the Deepfake Detection Challenge (DFDC). The goal of the challenge is to spur researchers around the world to build innovative new technologies that can help detect deepfakes and manipulated media.

Challenge participants must submit their code into a black box environment for testing. Participants will have the option to make their submission open or closed when accepting the prize. Open proposals will be eligible for challenge prizes as long as they abide by the open source licensing terms. Closed proposals will be proprietary and not be eligible to accept the prizes. Regardless of which track is chosen, all submissions will be evaluated in the same way. Results will be shown on the leaderboard.

The PAI Steering Committee has emphasized the need to ensure that all technical efforts incorporate attention to how the resulting code and products based on it can be made as accessible and useful as possible to key frontline defenders of information quality such as journalists and civic leaders around the world. The DFDC results will be a contribution to this effort and building a robust response to the emergent threat deepfakes pose globally.

