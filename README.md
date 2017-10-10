# CORSaliency
Corner-based Saliency Model
- Use to compute saliency map on a given image.
- An example is in the main function of CORS.py.

## Basic Idea
We proposed that a bottom-up visual saliency can be computed by directly detecting corners in a scene. Corners have the a lot of information as they defines shape, and human's eyes and brains are naturally attracted to them. 

Imagine how would you look at a line. Would you scan all parts linearly from the start to the end? Of course not.
What we usually do is just glancing at the starting and ending points, and we recognize the line and its length.

1. start --> ______________________ <-- 2. end

Not convinced how important corner information is yet? 
Try this thought experiment. You are just waking up, and after you open your eyes, all you can see is just plain white everywhere. Nothing, no objects, thus **no corners**. You might think, "Am I currently facing a big white wall?", and you'd turn around heavily in order to search for a line, or any evidence of a wall to validate this doubt. Or, you might think, "Am I happen to be floating in the air?", and you'd search for a skyline. All of this is drived by the urge to solve a nagging question, "where are you?". To make sense of a scene, we search for clues like objects, people, faces, lines, and they all share a common building block, **corners**.

While this idea might sound too simple, and yes there are flaws. However, its accuracy is competitive to other models, as shown by validating with [MIT300 dataset](http://saliency.mit.edu/results_mit300.html).


## Limitations
This idea only works in natural scenes, e.g., everyday photos that are taken with your camera. The model would perform badly in certain artificial scenes. Imagine a checker pattern, the algorithm will match each of the corners in an image, and thus predictions would scatter all over the image. However, a human would simply detect this repetition and stop paying attention to it. 

In this model, we addressed only the low-level bottom-up features. Thus, a viewer's internal goal and high-level pattern recognitions are not taken into account.

Contact: aun.rueopas@gmail.com
