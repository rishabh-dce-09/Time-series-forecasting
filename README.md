This code has been developed for the time series forecasting challenge at Kaggle.
Web link: https://www.kaggle.com/c/demand-forecasting-kernels-only


In case of graphviz issue, refer to the following link: https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft
As per the link, the steps to be followed to rectify the issue are:
  
  Step 1: Install Graphviz binary

    (a) Download Graphviz from http://www.graphviz.org/download/
    (b) Add below to PATH environment variable (mention the installed graphviz version):
      C:\Program Files (x86)\Graphviz2.38\bin
      C:\Program Files (x86)\Graphviz2.38\bin\dot.exe
    (d) Close any opened Juypter notebook and the command prompt
    (e) Restart Jupyter / cmd prompt and test
    
  Step 2: Install graphviz module for python (use either of 2)

    (a) pip
      pip install graphviz==0.18
    (b) conda
      conda install graphviz
