<h1 align="center">Hi 👋, I'm Thomas Kenison</h1>


<p align="center">
  <img src="https://komarev.com/ghpvc/?username=TomKenison162&label=Profile%20views&color=0e75b6&style=flat" alt="profile views" />
</p>

---


---


### 🛠️ My Tech Stack



#### **Frontend**
<p align="left"> 
    <a href="https://reactjs.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/react/react-original-wordmark.svg" alt="react" width="40" height="40"/> </a> 
    <a href="https://www.typescriptlang.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/typescript/typescript-original.svg" alt="typescript" width="40" height="40"/> </a> 
    <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/javascript/javascript-original.svg" alt="javascript" width="40" height="40"/> </a> 
    <a href="https://www.w3.org/html/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" width="40" height="40"/> </a> 
    <a href="https://www.w3schools.com/css/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/css3/css3-original-wordmark.svg" alt="css3" width="40" height="40"/> </a> 
    <a href="https://tailwindcss.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tailwindcss/tailwindcss-icon.svg" alt="tailwind" width="40" height="40"/> </a> 
</p>

#### **Backend**
<p align="left"> 
    <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> 
    <a href="https://www.java.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/java/java-original.svg" alt="java" width="40" height="40"/> </a> 
    <a href="https://dotnet.microsoft.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/dot-net/dot-net-original-wordmark.svg" alt="dotnet" width="40" height="40"/> </a> 
</p>

#### **Database & Caching**
<p align="left">
    <a href="https://www.postgresql.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/postgresql/postgresql-original-wordmark.svg" alt="postgresql" width="40" height="40"/> </a>
    <a href="https://www.mysql.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg" alt="mysql" width="40" height="40"/> </a>
</p>

#### **DevOps & Cloud**
<p align="left"> 
    <a href="https://www.docker.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original-wordmark.svg" alt="docker" width="40" height="40"/> </a> 
    <a href="https://kubernetes.io" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/kubernetes/kubernetes-plain-wordmark.svg" alt="kubernetes" width="40" height="40"/> </a> 
    <a href="https://azure.microsoft.com/en-us/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/azure/azure-original-wordmark.svg" alt="azure" width="40" height="40"/> </a>
    <a href="https://cloud.google.com" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg" alt="gcp" width="40" height="40"/> </a>
    <a href="https://git-scm.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/> </a> 
</p>

---

### 📊 My GitHub Stats

<p align="center">
    <br><br>
    <img align="center" src="https://github-readme-stats.vercel.app/api/top-langs?username=TomKenison162&show_icons=true&locale=en&layout=compact&theme=tokyonight" alt="TomKenison162's Top Languages" />
</p>

<p align="center">
    <img align="center" src="https://github-readme-streak-stats.herokuapp.com/?user=TomKenison162&theme=tokyonight" alt="TomKenison162's GitHub Streak" />
</p>
graph TD
    subgraph "Input & Preprocessing (train / predict)"
        X[<b>Input Data (X)</b><br>Shape: (n_samples, n_features)]
        Norm[<b>Standardization (Z-score)</b><br>X_norm = (X - self.mean) / self.std<br>Shape: (n_features, n_samples)]
        X -->|X.T| Norm
    end

    subgraph "Forward Pass (forward method)"
        Norm --> L1_Calc
        Params_W1b1(Params: W1, b1) --> L1_Calc
        L1_Calc(<b>Hidden Layer (Linear)</b><br>Z1 = W1 @ X_norm + b1<br>Shape: (32, n_samples))
        
        L1_Calc --> L1_Act
        L1_Act(<b>Hidden Layer (Activation)</b><br>A1 = ReLU(Z1)<br>Shape: (32, n_samples))
        
        L1_Act --> L2_Calc
        Params_W2b2(Params: W2, b2) --> L2_Calc
        L2_Calc(<b>Output Layer (Linear)</b><br>Z2 = W2 @ A1 + b2<br>Shape: (1, n_samples))
        
        L2_Calc --> L2_Act
        L2_Act(<b>Output Layer (Activation)</b><br>A2 = Sigmoid(Z2)<br>Shape: (1, n_samples))
    end

    subgraph "Output (predict method)"
        L2_Act --> Prob[<b>Probability (A2)</b><br>Value 0.0 to 1.0]
        Prob --> Pred[<b>Prediction</b><br>(A2 > 0.5)]
    end

    subgraph "Training (backward & optimise methods)"
        Y[<b>True Labels (Y)</b>] --> BP_L2
        L2_Act --> BP_L2
        BP_L2(<b>Backward (Layer 2)</b><br>dZ2 = A2 - Y (from Cross-Entropy Loss)<br>dW2, db2 (with L2 Reg))
        
        BP_L2 --> BP_L1
        L1_Act --> BP_L1
        Params_W2b2 --> BP_L1
        BP_L1(<b>Backward (Layer 1)</b><br>dZ1 = W2.T @ dZ2 * (Z1 > 0)<br>dW1, db1 (with L2 Reg))
        
        BP_L1 --> Optim
        Params_W1b1 --> Optim
        Params_W2b2 --> Optim
        Optim(<b>Update Parameters</b><br>Adam Optimizer<br>beta1=0.9, beta2=0.999)
    end
    
    %% Styles
    style X fill:#E0F7FA,stroke:#00796B
    style Norm fill:#B2EBF2,stroke:#00796B
    style L1_Calc fill:#E8F5E9,stroke:#388E3C
    style L1_Act fill:#C8E6C9,stroke:#388E3C
    style L2_Calc fill:#E8F5E9,stroke:#388E3C
    style L2_Act fill:#C8E6C9,stroke:#388E3C
    style Prob fill:#FFFDE7,stroke:#FBC02D
    style Pred fill:#FCE4EC,stroke:#D81B60
    style Y fill:#FFEBEE,stroke:#C62828
    style BP_L1 fill:#F3E5F5,stroke:#6A1B9A
    style BP_L2 fill:#F3E5F5,stroke:#6A1B9A
    style Optim fill:#EDE7F6,stroke:#4527A0
    style Params_W1b1 fill:#E1F5FE,stroke:#0277BD
    style Params_W2b2 fill:#E1F5FE,stroke:#0277BD
--- 

