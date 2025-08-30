import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import measure, color
import math as mt
import re
import os
import multiprocessing as mp
import sys
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import pandas as pd
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import inception_v3, resnext50_32x4d, Inception_V3_Weights, ResNeXt50_32X4D_Weights
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
import itertools
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
import joblib
from torch.nn.functional import softmax


cancel = False
class ImgDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(to_pil_image(image))
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def plot_acuracia(acuracia_treino=[
            0.3712, 0.3712, 0.3704, 0.3710, 0.3693,
            0.3704, 0.3686, 0.3675, 0.3725, 0.3719,
            0.3675, 0.3707, 0.3683, 0.3710, 0.3719
        ]):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(acuracia_treino) + 1), acuracia_treino, marker='o', color='blue')
        plt.title('Acurácia')
        plt.xlabel('Época')
        plt.ylabel('Acurácia de Treino')
        plt.grid(True)
        plt.show()

def get_filename(path):
    pos = 0
    if sys.platform == 'win32':
        pos = path.rfind('\\')
    elif sys.platform == 'linux' or sys.platform == 'darwin':
        pos = path.rfind('/')
    return path[pos+1:]

def process_clahe(img):
    clahe = cv.createCLAHE(clipLimit=5)
    clahe_img = np.clip(clahe.apply(img) + 20, 0, 255).astype(np.uint8)
    return clahe_img

def process_sobel(img):
    sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    sobel_img = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_img = np.uint8(np.clip(sobel_img, 0, 255))
    return sobel_img

def process_stretching(img):
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    return cv.LUT(img, table)

def show_img(original, processed):
    plt.subplot(1, 2, 1)  
    plt.imshow(original)  

    plt.title("Image 1") 
    plt.subplot(1, 2, 2)  
    plt.imshow(processed)  

    plt.title("Image 2") 
    plt.show()

def seg_otsu(img):
    _, thresh = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    return thresh

def abertura(img, kernel_dim=5, iterations = 1):
    kernel = np.ones((kernel_dim, kernel_dim), np.uint8) 
    img_open = cv.morphologyEx(img,cv.MORPH_OPEN,kernel, iterations = iterations)
    return img_open

def label_objects(img):

    label_image = measure.label(img, connectivity=1)  
    props = measure.regionprops(label_image)
    colored_label = color.label2rgb(label_image, image=img, bg_label=0)
    return colored_label, props

def mean_dist(props):
    min = 0xffffff
    mean = 0
    for i in props:
        for j in props:
            if(mt.dist(i.centroid,j.centroid)<min and j!=i):
                min =mt.dist(i.centroid,j.centroid)
        mean+=min

    return mean/len(props)

def get_props(props,data):
    m_area = 0
    m_circ = 0
    m_exc = 0
    m_radius = 0
    m_rad_dist = 0
    m_dis = mean_dist(props)
    tam = len(props)
    for i in props:
        m_area+=i.area
        m_circ+=pow(i.perimeter,2)/(4*np.pi*i.area)
        m_exc+=i.eccentricity
        m_radius+=i.equivalent_diameter_area/2
    m_area /= tam
    m_circ /= tam
    m_exc /= tam
    m_radius /= tam
    m_rad_dist = m_dis/m_radius
    return [get_filename(data[0]), m_area, m_circ, m_exc, m_dis, m_radius, m_rad_dist, tam, data[1]]

def pred_props(props):
    m_area = 0
    m_circ = 0
    m_exc = 0
    m_radius = 0
    m_rad_dist = 0
    m_dis = mean_dist(props)
    tam = len(props)
    for i in props:
        m_area+=i.area
        m_circ+=pow(i.perimeter,2)/(4*np.pi*i.area)
        m_exc+=i.eccentricity
        m_radius+=i.equivalent_diameter_area/2
    m_area /= tam
    m_circ /= tam
    m_exc /= tam
    m_radius /= tam
    m_rad_dist = m_dis/m_radius
    return [m_area, m_circ, m_exc, m_dis, m_radius, m_rad_dist, tam]

def rem_bg(img):
    img[img >= 240] = np.median(img[img<240])
    return img

def gray(img):
    return cv.cvtColor(img,cv.COLOR_BGR2GRAY)

def img_process(img):
    if type(img) == str:
        img = cv.imread(img)
    assert img is not None, "file could not be read, check with os.path.exists()"
    img = rem_bg(img)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    clahe = process_clahe(gray)
    otsu = seg_otsu(clahe)
    img_open = abertura(otsu, 5)
    labeled_img, props = label_objects(img_open)
    return labeled_img, props

def process_imgs(path_list, start, end, queue):
    result = []
    count = 0
    ini = time.time()
    for i in range(start,end):
        _, props = img_process(path_list[i][0])    
        result.append(get_props(props, path_list[i]))
        count+=1
        if(count%40 == 0):
            print("Elapsed: ", time.time()-ini)
            ini = time.time()
    queue.put(result)

def generate_csv(path_list,file):
    manager = mp.Manager()
    queue = manager.Queue()
    processes = []
    size = int(len(path_list)/mp.cpu_count())
    ranges = []
    x=0
    while(x<len(path_list)):
            ranges.append([x,x+size+1 if x+size< len(path_list) else len(path_list)])
            x+=size+1
    print(ranges)
    for start, end in ranges:
        p = mp.Process(target=process_imgs, args=(path_list, start, end, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    final_result = []
    while not queue.empty():
        final_result.extend(queue.get())

    with open (file,'w') as f:
        f.write('filename,area,circularidade,excentricidade,dis_media_nucleos,media_raio,raio_dis,qtde_objetos,label\n')
        for i in final_result:
            f.write(f"{i[0]},{i[1]},{i[2]},{i[3]},{i[4]},{i[5]},{i[6]},{i[7]},{i[8]}\n")
    print(len(final_result))

def train_xgb():
    train = pd.read_csv('train_xgb.csv')
    val = pd.read_csv('val_xgb.csv')
    X_train = train.drop(['label','filename','Patient ID'],axis=1)
    y_train = train['label']
    X_val = val.drop(['label','filename','Patient ID'],axis=1)
    y_val = val['label']
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_val, label=y_val)
    param = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'multi:softprob',
        'num_class': 3,
        'nthread': 4,
        'eval_metric': 'mlogloss',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 1
    }

    evallist = [(dtrain, 'train'), (dtest, 'test')]

    num_round = 300
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=num_round,
        evals=evallist,
        early_stopping_rounds=20,
        verbose_eval=True
    )

    y_pred_prob = bst.predict(dtest)

    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = y_val.values

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average='macro'))
    print(bst.get_score(importance_type='gain'))
    with open("model.pkl", "wb") as f:
        pickle.dump(bst, f)

def get_sets():
    df = pd.read_excel('patient-clinical-data.xlsx')
    original_dir = 'patches'
    labels = {
            'N0' : 0,
            'N+(1-2)' : 1,
            'N+(>2)' : 2
        }
    df['ALN status'] = df['ALN status'].map(labels)
    folders = sorted([f for f in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, f))])
    f_class = []

    for i in folders:
        f_class.append([i, df['ALN status'][int(i)-1]])

    f_class = pd.DataFrame(f_class, columns=['patient', 'label'])
    normal = f_class[f_class['label']==0]
    nv1 = f_class[f_class['label']==1]
    nv2 = f_class[f_class['label']==2]

    normal_folders = normal['patient'].tolist()
    nv1_folders = nv1['patient'].tolist()
    nv2_folders = nv2['patient'].tolist()
    train_normal, test_normal = train_test_split(normal_folders, random_state=42, test_size=0.2,shuffle=True)
    train_nv1, test_nv1 = train_test_split(nv1_folders, random_state=42, test_size=0.2,shuffle=True)
    train_nv2, test_nv2 = train_test_split(nv2_folders, random_state=42, test_size=0.2,shuffle=True)

    train_normal, val_normal = train_test_split(train_normal, random_state=42, test_size=0.25,shuffle=True)
    train_nv1, val_nv1 = train_test_split(train_nv1, random_state=42, test_size=0.25,shuffle=True)
    train_nv2, val_nv2 = train_test_split(train_nv2, random_state=42, test_size=0.25,shuffle=True)
    train_set = train_normal+train_nv1+train_nv2
    val_set = val_normal+val_nv1+val_nv2
    test_set = test_normal+test_nv1+test_nv2
    return train_set, val_set, test_set

def xgb_split():
    train_set,val_set,test_set = get_sets()
    with (
        open('props_concat.csv', 'r') as data,
        open('train_xgb.csv', 'w') as f1,
        open('val_xgb.csv', 'w') as f2,
        open('test_xgb.csv', 'w') as f3
    ):
        f1.write('Patient ID,Age(years),Tumour Size(cm),ER,PR,HER2,HER2 Expression,Surgical,Molecular subtype_HER2(+),Molecular subtype_Luminal A,Molecular subtype_Luminal B,Molecular subtype_Triple negative,Tumour Type_Invasive ductal carcinoma,Tumour Type_Invasive lobular carcinoma,Tumour Type_Other type,filename,area,circularidade,excentricidade,dis_media_nucleos,media_raio,raio_dis,qtde_objetos,label\n')
        f2.write('Patient ID,Age(years),Tumour Size(cm),ER,PR,HER2,HER2 Expression,Surgical,Molecular subtype_HER2(+),Molecular subtype_Luminal A,Molecular subtype_Luminal B,Molecular subtype_Triple negative,Tumour Type_Invasive ductal carcinoma,Tumour Type_Invasive lobular carcinoma,Tumour Type_Other type,filename,area,circularidade,excentricidade,dis_media_nucleos,media_raio,raio_dis,qtde_objetos,label\n')
        f3.write('Patient ID,Age(years),Tumour Size(cm),ER,PR,HER2,HER2 Expression,Surgical,Molecular subtype_HER2(+),Molecular subtype_Luminal A,Molecular subtype_Luminal B,Molecular subtype_Triple negative,Tumour Type_Invasive ductal carcinoma,Tumour Type_Invasive lobular carcinoma,Tumour Type_Other type,filename,area,circularidade,excentricidade,dis_media_nucleos,media_raio,raio_dis,qtde_objetos,label\n')
        data.readline()
        for line in data:
            pid = re.findall(r"^\s*(\d+)",line)[0]

            if(pid in train_set):
                f1.write(line)
            elif(pid in val_set):
                f2.write(line) 
            elif(pid in test_set):
                f3.write(line)
    print("Completo")

def data_split(original_dir = 'patches'):
    df = pd.read_excel('patient-clinical-data.xlsx')
    labels = {
            'N0' : 0,
            'N+(1-2)' : 1,
            'N+(>2)' : 2
        }
    df['ALN status'] = df['ALN status'].map(labels)
    folders = sorted([f for f in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, f))])
    f_class = []

    for i in folders:
        f_class.append([i, df['ALN status'][int(i)-1]])

    f_class = pd.DataFrame(f_class, columns=['patient', 'label'])
    normal = f_class[f_class['label']==0]
    nv1 = f_class[f_class['label']==1]
    nv2 = f_class[f_class['label']==2]

    normal_folders = normal['patient'].tolist()
    nv1_folders = nv1['patient'].tolist()
    nv2_folders = nv2['patient'].tolist()

    train_normal, test_normal = train_test_split(normal_folders, random_state=42, test_size=0.2,shuffle=True)
    train_nv1, test_nv1 = train_test_split(nv1_folders, random_state=42, test_size=0.2,shuffle=True)
    train_nv2, test_nv2 = train_test_split(nv2_folders, random_state=42, test_size=0.2,shuffle=True)
    print(len(train_nv2))

    train_normal, val_normal = train_test_split(train_normal, random_state=42, test_size=0.25,shuffle=True)
    train_nv1, val_nv1 = train_test_split(train_nv1, random_state=42, test_size=0.25,shuffle=True)
    train_nv2, val_nv2 = train_test_split(train_nv2, random_state=42, test_size=0.25,shuffle=True)
    print(len(train_nv2))
    with open('train_labels.csv',"w") as f:
        for split in [train_normal,train_nv1,train_nv2]:
            for folder in split:
                for root, _, files in os.walk(os.path.join(original_dir,folder)):
                    for img in files:
                        if img.endswith(".jpg"):
                            full_path = os.path.join(folder,img)
                            label = df['ALN status'][int(folder)-1]
                            f.write(f"{full_path},{label}\n")

    with open('test_labels.csv',"w") as f:
        for split in [test_normal,test_nv1,test_nv2]:
            for folder in split:
                for root, _, files in os.walk(os.path.join(original_dir,folder)):
                    for img in files:
                        if img.endswith(".jpg"):
                            full_path = os.path.join(folder,img)
                            label = df['ALN status'][int(folder)-1]
                            f.write(f"{full_path},{label}\n")

    with open('val_labels.csv',"w") as f:
        for split in [val_normal,val_nv1,val_nv2]:
            for folder in split:
                for root, _, files in os.walk(os.path.join(original_dir,folder)):
                    for img in files:
                        if img.endswith(".jpg"):
                            full_path = os.path.join(folder,img)
                            label = df['ALN status'][int(folder)-1]
                            f.write(f"{full_path},{label}\n")
    print("\nCompleto")

def train(folder, sel, epochs=10):
    global cancel
    best_val_acc = 0
    accs = []
    model = None
    transform = None
    batch_size = 0

    if(sel == 0):
        model = inception_v3(weights=Inception_V3_Weights)
        transform = Compose([
        Resize((299, 299)),  
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 3)
        batch_size = 128
    elif(sel == 1):
        model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights)
        transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        batch_size= 128

    train_data = ImgDataset('train_labels.csv',folder,transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = ImgDataset('val_labels.csv',folder,transform=transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    ini = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        for inputs, labels in train_loader:
            if(cancel):
                print("Limpando memória")
                del model
                del optimizer
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                cancel = False
                print("Treino Cancelado")
                return
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if sel == 0:
                torch.save(model.state_dict(), 'best_model_inc.pth')
            elif sel == 1:
                torch.save(model.state_dict(), 'best_model_rex.pth')
        accs.append(epoch_acc)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")
    plot_acuracia(accs)
    if sel == 0:
        torch.save(model.state_dict(), 'last_model_inc.pth')
    elif sel == 1:
        torch.save(model.state_dict(), 'last_model_rex.pth')        
    print("Elapsed: ", time.time()-ini)

def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = correct / total
    return acc

def concat_csvs(file,queue):
    df=pd.read_excel('patient-clinical-data.xlsx').drop(['Number of lymph node metastases','ALN status'],axis=1)
    scaler = MinMaxScaler()
    le = LabelEncoder()
    ohe = OneHotEncoder(sparse_output=False)
    df['Age(years)'] = scaler.fit_transform(df[['Age(years)']])
    joblib.dump(scaler,'scaler.save')

    binary = []
    for i in df.columns:
        if len(df[i].unique()) == 2:
            binary.append(i)
    for i in binary:
        df[i] = le.fit_transform(df[i])
    values = {
            '0' : 0,
            '1+' : 1,
            '2+' : 2,
            '3+' : 3
        }
    df['HER2 Expression'] = df['HER2 Expression'].map(values)
    molecular = pd.DataFrame(ohe.fit_transform(df[['Molecular subtype']]),columns=ohe.get_feature_names_out(['Molecular subtype']))
    df = pd.concat([df.drop('Molecular subtype',axis=1),molecular],axis=1)
    joblib.dump(ohe, 'molecular_encoder.save')
    tumour = pd.DataFrame(ohe.fit_transform(df[['Tumour Type']]),columns=ohe.get_feature_names_out(['Tumour Type']))
    df = pd.concat([df.drop('Tumour Type',axis=1),tumour],axis=1)
    df = df.drop(['Histological grading', 'Ki67'],axis=1)
    joblib.dump(ohe, 'tumour_encoder.save')
    base = pd.read_csv(file)
    to_concat = []
    for _, i in base.iterrows():
        patient = int(re.findall(r"^\d+",i['filename'])[0])
        to_concat.append(df.iloc[patient-1].T)
    to_concat = pd.DataFrame(to_concat,columns=df.columns)
    to_concat = to_concat.reset_index(drop=True)
    base = pd.concat([to_concat,base],axis=1)
    base.to_csv('props_concat.csv',index=False)
    queue.put(21)
    print('Completo')

def scatter():
    df = pd.read_csv('props_concat.csv')
    cor_classes = {
        0: 'black',
        1: 'blue',
        2: 'red'
    }
    features = ['area', 'circularidade', 'excentricidade', 'dis_media_nucleos', 'media_raio', 'raio_dis','qtde_objetos']
    pares = list(itertools.combinations(features, 2))

    plt.figure(figsize=(18, 15))

    for i, (x, y) in enumerate(pares):
        plt.subplot(5, 5, i + 1)  
        for classe, cor in cor_classes.items():
            subset = df[df['label'] == classe]
            plt.scatter(subset[x], subset[y], color=cor, label=classe, alpha=0.7)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle("Gráficos de Dispersão das Características aos Pares", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()

def classify(path, img, sel):
    weights = torch.load(path)
    model = None
    transform = None
    if sel == 0:
        transform = Compose([
        Resize((299, 299)),  
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        model = inception_v3(weights=None)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 3)
        model.fc = nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(weights)
    elif sel == 1:
        transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model = resnext50_32x4d(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(weights)
    model.eval()
    img = transform(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        return torch.argmax(softmax(output))

def cancel_train():
    global cancel
    cancel = True

def test_xgb():
    model = None
    all_preds = []
    all_labels = []
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    data = pd.read_csv('test_xgb.csv').drop(['Patient ID','filename'],axis=1)
    X_test = data.drop(['label'],axis=1)
    y_test = data['label']
    dtest = xgb.DMatrix(X_test, label=y_test)
    test_prob = model.predict(dtest)
    all_labels = y_test.values
    all_preds = np.argmax(test_prob, axis=1)
    return [all_labels,all_preds]

def plot_metrics(all_labels, all_preds):
    acc = accuracy_score(all_labels, all_preds)
    print(f'Acurácia no conjunto de teste: {acc:.4f}')
    cm = confusion_matrix(all_labels, all_preds)
    n_classes = cm.shape[0]
    especificidade = []
    sensibilidade = []
    for i in range(n_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        sens = TP / (TP + FN) if (TP + FN) != 0 else 0
        esp  = TN / (TN + FP) if (TN + FP) != 0 else 0
        sensibilidade.append(round(sens,4))
        especificidade.append(round(esp,4))
        print(f"Classe {i}:")
        print(f"  Sensibilidade: {sens:.4f}")
        print(f"  Especificidade: {esp:.4f}")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.text(0.5, -0.2, f'Acurácia: {acc:.4f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.subplots_adjust(bottom=0.2)
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 5)) 
    ax.axis('off') 

    row_labels = ['Especificidade', 'Sensibilidade']
    col_labels = ['Classe 0', 'Classe 1', 'Classe 2']
    table_vals = [especificidade, sensibilidade]

    tabela = ax.table(cellText=table_vals, 
                    rowLabels=row_labels, 
                    colLabels=col_labels, 
                    loc='center')

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(12)
    tabela.scale(1.2,1.2)
    fig.suptitle('Métricas por Classe', fontsize=14) 
    plt.subplots_adjust(left=0.3)
    plt.show()

def teste(weights,folder,sel):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    transform = None
    batch_size = 0
    if(sel == 0):
        model = inception_v3(weights=None)
        transform = Compose([
        Resize((299, 299)),  
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 3)
        batch_size = 128
    elif(sel == 1):
        model = resnext50_32x4d(weights=None)
        transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        batch_size= 128
    test_data = ImgDataset('test_labels.csv',folder,transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)        
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(weights, map_location=device))
    model = model.to(device)

    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
                
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return [all_labels,all_preds]