# supervised_learning_pacman
Pacman trained with supervised learning using UC Berkeley CS188 Intro to AI Pacman's implemetation.

To run the code, please first pip install the following dependencies:
  - pip3 install future
  - pip3 install tkinter
  - pip3 install javabridge
  - pip3 install python-weka-wrapper3

Some errors may happen when running in linux, so windows is recommended. However, javabridge is noticeably problematic to install, beware.

Finally, acces the folder "pacman" and run the following command:

``
python busters.py -p WekaBustersAgent
``

You can chang the layout by adding the -l and the name of any layout found in the folder layouts.
You can also make the ghosts move randomly by adding -g RandomGhost .
