# -*- coding: utf-8 -*-
import os
import  pandas as pd
import numpy as np
import random
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from scipy.stats import kstest,ttest_ind
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# 文物类
class relic():
    def __init__(self,numberring,waypath = '附件.xlsx'):
        self.sheet = ['表单1','表单2','表单3']
        self.content = [pd.read_excel(waypath,sheet_name=i,) for i in self.sheet ]
        self.number = numberring
        self.type = self.content[0][self.content[0]['文物编号']==self.number]['类型'].values[0]
        if self.content[0][self.content[0]['文物编号']==self.number]['表面风化'].values[0] == '风化':
            self.fy = 1
        else:
            self.fy = 0
        self.useful_index = self.find_all()
        data_worker =  self.data_processor()
        self.data = data_worker[0]
        self.kmeans_data = data_worker[1]
    '''
        添加可用序列索引
    # 不添加风化文物的未风化部分索引 ,比如23
    '''
    def find_all(self):
        need_column = self.content[1]['文物采样点'].values
        useful_index = []
        special_index = [] # 存放有风化的无风化部分的索引
        serious_damage_index = [] # 存放严重风化部分的索引
        for need_list,i in zip(need_column,range(len(need_column))):
            number = need_list[:2]
            
            if len(need_list) > 2 and int(number) == self.number:
                fy_part_yes_or_no = need_list[2:]
                if (self.fy == 1) and (fy_part_yes_or_no.find("未风化")==-1) and (fy_part_yes_or_no.find('严重')==-1):
                    # print("111111")
                    useful_index.append(i)
                # 无风化的直接录入
                elif self.fy == 0:
                    useful_index.append(i)
                elif self.fy == 1 and fy_part_yes_or_no.find("未风化") != -1 :
                    special_index.append(i)
                else :
                    serious_damage_index.append(i)
                    useful_index.append(i)
            elif len(need_list) == 2 and int(number) == self.number:
                useful_index.append(i)
        self.special_index = special_index
        self.serious_damage_index = serious_damage_index
        return useful_index
    # 根据索引取数据，并先进行数据处理
    def  data_processor(self):
        all_data = self.content[1][self.content[1].columns[1:]].fillna(0)
        this_data = []
        kmeans_data = []
        if(len(self.useful_index) != 0 or len(self.special_index)):
            for i in self.useful_index:
                temp = all_data.loc[i].values
                # 处理数据使得变为100%
                sum_ = 0
                for value in temp:
                    sum_ += value
                # 乘子
                muliply_num = 100.0/sum_
                temp = [i*muliply_num for i in temp]
                this_data.append(temp)
                kmeans_data.append(temp)
            for i in self.special_index:
                temp = all_data.loc[i].values
                # 处理数据使得变为100%
                sum_ = 0
                for value in temp:
                    sum_ += value
                # 乘子
                muliply_num = 100.0/sum_
                temp = [i*muliply_num for i in temp]
                this_data.append(temp)
                # kmeans_data.append(temp)
            for i in self.serious_damage_index:
                temp = all_data.loc[i].values
                # 处理数据使得变为100%
                sum_ = 0
                for value in temp:
                    sum_ += value
                # 乘子
                muliply_num = 100.0/sum_
                temp = [i*muliply_num for i in temp]
                kmeans_data.append(temp)
            return [this_data,kmeans_data]
        else :
            # print('this num = %d has no useful index'%self.number)
            return [this_data,kmeans_data]
        
# 导入数据
class dataloader():
    def __init__(self,waypath = '附件.xlsx',train = True,model_print = True,enrich_yes_or_no = True):
        self.waypath = waypath
        self.sheet = ['表单1','表单2','表单3']
        self.content = [pd.read_excel(waypath,sheet_name=i,) for i in self.sheet ]
        origin_data = self.content[1][self.content[1].columns[1:]].fillna(0)
        self.columns_names = origin_data.columns.values
        og_data = self.get_kinds_data()
        self.all_data = og_data[0]
        self.high_k_fy_data = og_data[1]
        self.high_k_nfy_data = og_data[2]
        self.pbba_fy_data = og_data[3]
        self.pbba_nfy_data = og_data[4]
        self.origin_high_k_fy_data = og_data[1]
        self.origin_high_k_nfy_data = og_data[2]
        self.origin_pbba_fy_data = og_data[3]
        self.origin_pbba_nfy_data = og_data[4]
        param = self.gauss_param_get()
        
        self.hkf_param = param[0] # 高钾有风化
        self.hknf_param = param[1]# 高钾无风化
        self.pbf_param = param[2]# 铅钡有风化
        self.pbnf_param = param[3]# 铅钡无风化
        self.first_enrich = True
        test_data = self.enrich_dataset(enrich_yes_or_no) # 丰富数据集
        self.hk_test_data = test_data[0]
        self.pb_test_data = test_data[1]
        if(train == True):
            model = self.multi_linear(model_print) # 多元线性回归
            self.hk_model = model[0]
            self.pb_model = model[1]
    def relative_judge(self,use_chi = True):
        # 认为缺失颜色信息的数据无用，删除
        data = self.content[0]
        no_null_data = data.dropna()
        columns = ['纹饰','类型','颜色']
        fy_columns = ['无风化','风化']
        kinds = [list(set(no_null_data[i])) for i in columns] 
        
        # 卡方检验,运行结果中的P值就是我们认为的可信度
        if use_chi:
            for detail_kinds,i in zip(kinds,range(len(kinds))):
                print('---------------------------------------------------')
                print(detail_kinds)
                # is_fy中保存对应的类型的风化数目
                # isnot_fy中保存对应类型的未风华数目
                is_fy = []
                isnot_fy = []
                for type_in in detail_kinds:
                    index = no_null_data[columns[i]].loc[no_null_data[columns[i]].str.contains(type_in)].index.tolist()
                    is_fy_counter = 0
                    isnot_fy_counter = 0
                    for count in range(len(index)):
                        if(no_null_data.loc[index[count],'表面风化'] == '风化'):
                            is_fy_counter+=1
                        else:
                            isnot_fy_counter+=1
                    is_fy.append(is_fy_counter)
                    isnot_fy.append(isnot_fy_counter)
                df = pd.DataFrame([is_fy,isnot_fy],index = fy_columns ,columns = detail_kinds)
                kt = chi2_contingency(df)
                print('卡方值=%.4f, p值=%.4f, 自由度=%i 期望_frep=%s'%kt)
                print('---------------------------------------------------')
    def get_kinds_data(self):
        all_data = []
        high_k_fy_data = []
        high_k_nfy_data = []
        pbba_fy_data = []
        pbba_nfy_data = []
        for i in range(1,59):
            this_relic = relic(i)
            if(len(this_relic.useful_index) != 0 or len(this_relic.special_index) !=0):
                all_data.extend(this_relic.data)
                if(this_relic.type == '高钾' and this_relic.fy == 1):
                    high_k_fy_data.extend(this_relic.data)
                elif(this_relic.type == '高钾' and this_relic.fy == 0):
                    high_k_nfy_data.extend(this_relic.data)
                elif(this_relic.type == '铅钡' and this_relic.fy == 1):
                    pbba_fy_data.extend(this_relic.data)
                elif(this_relic.type == '铅钡' and this_relic.fy == 0):
                    pbba_nfy_data.extend(this_relic.data)
            else:
                print("!!!!!!!!!!!")
        return [np.mat(all_data),
                np.mat(high_k_fy_data),
                np.mat(high_k_nfy_data),
                np.mat(pbba_fy_data),
                np.mat(pbba_nfy_data)]
    # 认为这些组元的数据是一个正太分布，计算它们的μ和σ,基于此我们可以生成需要的无风化数据使用
    def gauss_param_get(self):
        
        # 数据不选其中风化中未风化的部分和严重风化的部分
        high_k_fy_miu_list = []
        high_k_fy_sigma_list = []
        pbba_fy_miu_list = []
        pbba_fy_sigma_list = []
        high_k_nfy_miu_list = []
        high_k_nfy_sigma_list = []
        pbba_nfy_miu_list = []
        pbba_nfy_sigma_list = []
        all_data = self.all_data
        high_k_fy_data = self.high_k_fy_data
        high_k_nfy_data = self.high_k_nfy_data
        pbba_fy_data = self.pbba_fy_data
        pbba_nfy_data = self.pbba_nfy_data
        for this_name,i in zip(self.columns_names,range(len(self.columns_names))):
            
          #  miu = np.array(all_data[:,i]).mean()
          # sigma = np.array(all_data[:,i]).std()
            hkf_miu = high_k_fy_data[:,i].mean()
            hkf_sigma = high_k_fy_data[:,i].std()
            hknf_miu = high_k_nfy_data[:,i].mean()
            hknf_sigma = high_k_nfy_data[:,i].std()
            pbf_miu = pbba_fy_data[:,i].mean()
            pbf_sigma = pbba_nfy_data[:,i].std()
            pbnf_miu = pbba_nfy_data[:,i].mean()
            pbnf_sigma = pbba_nfy_data[:,i].std()
            # print(miu,sigma)
            high_k_fy_miu_list.append(hkf_miu)
            high_k_fy_sigma_list.append(hkf_sigma)
            high_k_nfy_miu_list.append(hknf_miu)
            high_k_nfy_sigma_list.append(hknf_sigma)
            pbba_fy_miu_list.append(pbf_miu)
            pbba_fy_sigma_list.append(pbf_sigma)
            pbba_nfy_miu_list.append(pbnf_miu)
            pbba_nfy_sigma_list.append(pbnf_sigma)
            '''
            # kstest 检查
            ks_result = kstest(all_data[:,i],'norm',(miu,sigma))
            print(all_data[:,0])
            if(ks_result.pvalue > 0.05):
                print("ths %s column suits gauss func"%(this_name))
            '''
        return [[high_k_fy_miu_list,high_k_fy_sigma_list],
                [high_k_nfy_miu_list,high_k_nfy_sigma_list],
                [pbba_fy_miu_list,pbba_fy_sigma_list],
                [pbba_fy_miu_list,pbba_nfy_sigma_list]]
        # return [miu_list,sigma_list]
        # print(columns_names)
    # 利用计算出的参数来丰富数据集合，注意，只要有风化的数据，预测得到无风化时的化学成分
    # 由于SiO2为玻璃的主要成分，按照SiO2的含量指定关系，从高到低排序,
    def enrich_dataset(self,enrich_yes_or_no = True):
        # print(self.hkf_param)
        hkf_miu = np.mat(self.hkf_param[0])
        hkf_sigma = np.mat(self.hkf_param[1])
        hknf_miu = np.mat(self.hknf_param[0])
        hknf_sigma = np.mat(self.hknf_param[1])
        pbf_miu = np.mat(self.pbf_param[0])
        pbf_sigma = np.mat(self.pbf_param[1])
        pbnf_miu = np.mat(self.pbnf_param[0])
        pbnf_sigma = np.mat(self.pbf_param[1])
        # 扩充30个数据
        self.origin_high_k_fy_data = self.high_k_fy_data
        self.origin_pbba_nfy_data = self.pbba_nfy_data
        self.origin_high_k_nfy_data = self.high_k_nfy_data
        self.origin_pbba_fy_data = self.pbba_fy_data
        if enrich_yes_or_no == True:
            for i  in range(30):
                random_number = random.uniform(-2, 2) #数据的生成符合3σ原则
                new_hkf_data = hkf_miu + random_number*hkf_sigma
                new_hknf_data = hknf_miu + random_number*hknf_sigma
                new_pbf_data = pbf_miu + random_number*pbf_sigma
                new_pbnf_data = pbnf_miu + random_number*pbnf_sigma
                self.high_k_fy_data = np.row_stack([self.high_k_fy_data,new_hkf_data])
                self.high_k_nfy_data = np.row_stack([self.high_k_nfy_data,new_hknf_data])
                self.pbba_fy_data = np.row_stack([self.pbba_fy_data,new_pbf_data])
                self.pbba_nfy_data = np.row_stack([self.pbba_nfy_data,new_pbnf_data])
            
            if self.first_enrich == True:
                for i in range(23):
                    random_number = random.uniform(-2, 2)
                    new_pbnf_data = pbnf_miu + random_number*pbnf_sigma
                    self.pbba_nfy_data = np.row_stack([self.pbba_nfy_data,new_pbnf_data])
                for i in range(8):
                    random_number = random.uniform(-2, 2)
                    new_hkf_data = hkf_miu + random_number*hkf_sigma
                    self.high_k_fy_data = np.row_stack([self.high_k_fy_data,new_hkf_data])
                '''for i in range(3):
                    random_number = random.uniform(-2, 2)
                    new_pbf_data = pbf_miu + random_number*pbf_sigma
                    self.pbba_fy_data = np.row_stack([self.pbba_fy_data,new_pbf_data])'''
                self.high_k_fy_data = self.high_k_fy_data[np.lexsort([-self.high_k_fy_data.T[0]])][0]
                self.high_k_nfy_data = self.high_k_nfy_data[np.lexsort([-self.high_k_nfy_data.T[0]])][0]
                self.pbba_fy_data = self.pbba_fy_data[np.lexsort([-self.pbba_fy_data.T[0]])][0]
                self.pbba_nfy_data = self.pbba_nfy_data[np.lexsort([-self.pbba_nfy_data.T[0]])][0]
                self.first_enrich = False
            
        # 生成测试集
        hkf_test_data = []
        hknf_test_data = []
        pbf_test_data = []
        pbnf_test_data = []
        for i in range(30):
            random_number = random.uniform(-2, 2)
            new_hkf_data = hkf_miu + random_number*hkf_sigma
            new_hknf_data = hknf_miu + random_number*hknf_sigma
            new_pbf_data = pbf_miu + random_number*pbf_sigma
            new_pbnf_data = pbnf_miu + random_number*pbnf_sigma
            if i == 0:
                hkf_test_data = new_hkf_data
                hknf_test_data = new_hknf_data
                pbf_test_data = new_pbf_data
                pbnf_test_data = new_pbnf_data
            else:
                hkf_test_data = np.row_stack([hkf_test_data,new_hkf_data])
                hknf_test_data = np.row_stack([hknf_test_data,new_hknf_data])
                pbf_test_data = np.row_stack([pbf_test_data,new_pbf_data])
                pbnf_test_data = np.row_stack([pbnf_test_data,new_pbnf_data])
        # print(hkf_test_data)
        '''hkf_test_data = np.mat(hkf_test_data)
        hknf_test_data = np.mat(hknf_test_data)
        pbf_test_data = np.mat(pbf_test_data)
        pbnf_test_data = np.mat(pbnf_test_data)'''
        
        hkf_test_data = hkf_test_data[np.lexsort([-hkf_test_data.T[0]])][0]
        hknf_test_data = hknf_test_data[np.lexsort([-hknf_test_data.T[0]])][0]
        pbf_test_data = pbf_test_data[np.lexsort([-pbf_test_data.T[0]])][0]
        pbnf_test_data = pbnf_test_data[np.lexsort([-pbnf_test_data.T[0]])][0]
        
        return [[hknf_test_data,hkf_test_data],
                [pbnf_test_data,pbf_test_data]]
        # print(self.high_k_nfy_data.sort(key=self.matrix_sort()))
        # enrich 完成之后按照要求排序
        #print(type(new_hkf_data),type(self.high_k_fy_data[0]))
        
       # print(self.high_k_nfy_data)
    # 多元线性回归
    def multi_linear(self,model_print = True):
        hk_model = LinearRegression()
        pb_model = LinearRegression()
       #  print(self.high_k_nfy_data)
       #  print(self.high_k_nfy_data.shape)
       # print(len(self.high_k_fy_data),len(self.high_k_nfy_data))
       # print(len(self.pbba_fy_data),len(self.pbba_nfy_data))
        
        #print(self.high_k_nfy_data)
        #print(self.pbba_nfy_data)
        
        hk_model.fit(self.high_k_fy_data,self.high_k_nfy_data)
        pb_model.fit(self.pbba_fy_data,self.pbba_nfy_data)
        hk_test_result = hk_model.predict(self.hk_test_data[1])
        pb_test_result = pb_model.predict(self.pb_test_data[1])
        hk_coef_ = hk_model.coef_
        hk_intercept = hk_model.intercept_
        pb_coef_ = pb_model.coef_
        pb_intercept = pb_model.intercept_
        if(model_print):
            print('高钾类:')
            print('MSE = %f'%mean_squared_error(self.hk_test_data[0], hk_test_result))
            print('MAE = %f'%mean_absolute_error(self.hk_test_data[0],hk_test_result))
            print('R2 = %f'%r2_score(self.hk_test_data[0],hk_test_result))
            #print("系数：",hk_coef_,"\n常数项",hk_intercept)
            print('铅钡类:')
            print('MSE = %f'%mean_squared_error(self.pb_test_data[0], pb_test_result))
            print('MAE = %f'%mean_squared_error(self.pb_test_data[0], pb_test_result))
            print('R2 = %f'%r2_score(self.pb_test_data[0],pb_test_result))
            # origin_data_
            if(r2_score(self.hk_test_data[0],hk_test_result)>0.9):
                writer = pd.ExcelWriter('data_sheet.xlsx')
                counter = 0
                predict_hkf_data = hk_model.predict(self.origin_high_k_fy_data)
                predict_pbf_data = pb_model.predict(self.origin_pbba_fy_data)
                for sy in  range(4):
                    counter += 1
                    if sy == 0:
                        data = pd.DataFrame(self.origin_high_k_fy_data)
                        data.columns = self.columns_names
                    elif sy == 1:
                        data = pd.DataFrame(self.origin_pbba_fy_data)
                        data.columns = self.columns_names
                    elif sy == 2:
                        data = pd.DataFrame(predict_hkf_data)
                        data.columns = self.columns_names
                    else:
                        data = pd.DataFrame(predict_pbf_data)
                        data.columns = self.columns_names
                    data.to_excel(writer,'sheet'+str(counter))
                writer.save()
                writer.close()
            
            #print("系数：",pb_coef_,"\n常数项",pb_intercept)
        # print(hk_model)
        # print(pb_model)
        return [hk_model,pb_model]
# 针对问题二的第一个聚类模型
# 思路：分为有风化和无风化之后分别聚类为高钾和铅钡两种，并和真实数据进行比较
class kmeans_solver_1():
    def __init__(self):
        data = self.get_kinds_data()
        self.fy_part = data[0]
        self.fy_part_true_kind = data[2]
        self.no_fy_part_true_kind = data[3]
        self.no_fy_part = data[1]
        labels = self.let_kmeans()
        self.fy_train_labels = labels[0]
        self.no_fy_train_labels = labels[1]
        rate = self.evaluation()
        self.fy_Acc_rate = rate[0]
        self.nfy_Acc_rate = rate[1]
    def get_kinds_data(self):
        fy_part = []
        no_fy_part = []
        fy_part_true_kind = []
        no_fy_part_true_kind = []
        for i in range(1,59):
            this_relic  = relic(i)
            if this_relic.fy == 1:
                if len(this_relic.data)==0:
                    # print("!!!!!")
                    continue
                else:
                    for i in range(len(this_relic.kmeans_data)):
                        fy_part_true_kind.append(this_relic.type)
                    fy_part.extend(this_relic.kmeans_data)
                # fy_part.append(this_relic)
            else:
                no_fy_part.extend(this_relic.kmeans_data)
                for i in range(len(this_relic.kmeans_data)):
                    no_fy_part_true_kind.append(this_relic.type)
                # no_fy_part.append(this_relic)
        return [fy_part,no_fy_part,fy_part_true_kind,no_fy_part_true_kind]
    def let_kmeans(self):
        fy_kmeans = KMeans(n_clusters=2,random_state=0).fit(self.fy_part)
        no_fy_kmeans = KMeans(n_clusters=2,random_state=0).fit(self.no_fy_part)
        fy_train_labels = fy_kmeans.labels_
        no_fy_train_labels = no_fy_kmeans.labels_
        
        print('labels: \n',fy_train_labels)
        print(self.fy_part_true_kind)
        print('labels: \n',no_fy_train_labels)
        print(self.no_fy_part_true_kind)
        return [fy_train_labels,no_fy_train_labels]
        
    def evaluation(self):
        correct_fy = 0.0
        correct_nfy = 0.0
        for i,j in zip(self.fy_part_true_kind,self.fy_train_labels):
            if (i == '高钾' and j == 0) or (i == '铅钡' and j == 1):
                correct_fy += 1
        fy_Acc_rate = correct_fy/len(self.fy_part_true_kind)
        for i,j in zip(self.no_fy_part_true_kind,self.no_fy_train_labels):
            if (i == '高钾' and j == 0) or (i == '铅钡' and j == 1):
                correct_nfy +=1
        nfy_Acc_rate = correct_nfy/len(self.no_fy_train_labels)
        print('有风化的正确率为：%f\n无风化的正确率为:%f'%(fy_Acc_rate,nfy_Acc_rate))
        return [fy_Acc_rate,nfy_Acc_rate]
        
# 第二题第二问 亚类划分 使用无风化的数据作为输入
class kmeans_solver_2():
    def __init__(self):
        got_data = self.get_data()
        self.hk_data = got_data[0]
        self.pb_data = got_data[1]
        pca_data = self.params_change()
        self.hk_pca_data = pca_data[0]
        self.pb_pca_data = pca_data[1]
        pass
    def get_data(self):
        loader = dataloader(train = True,model_print=False,enrich_yes_or_no=True)
        self.loader = loader
        # loader.enrich_dataset()
        # 选择无风化的部分作为输入
        hk_data = np.row_stack([np.mat(loader.origin_high_k_nfy_data),loader.hk_model.predict(loader.origin_high_k_fy_data)])
        pb_data = np.row_stack([np.mat(loader.origin_pbba_nfy_data),loader.pb_model.predict(loader.origin_pbba_fy_data)])
        print(hk_data.shape,pb_data.shape,np.mat(loader.high_k_nfy_data).shape,)
        #hk_data = loader.hk_test_data[0]
        #pb_data = loader.pb_test_data[0]
        # print(len(hk_data))
        # print(len(pb_data))
        return [hk_data,pb_data]
    def params_change(self,param = 0.95):
        hk_data = self.hk_data
        pb_data = self.pb_data
        hk_pca = PCA(n_components= param)
        hk_pca.fit(hk_data)
        pb_pca = PCA(n_components= param)
        pb_pca.fit(pb_data)
        # hk_pca_data = hk_pca.transform()
        # plt.plot(hk_pca.explained_variance_)
        # plt.plot(pb_pca.explained_variance_)
        # plt.legend(['hk_pca','pb_pca'])
        #plt.show()
        hk_pca_data = hk_pca.fit_transform(hk_data)
        pb_pca_data = pb_pca.fit_transform(pb_data)
        return [hk_pca_data,pb_pca_data]
        # print(hk_data)
        pass
    def plot_show(self):
        print(self.hk_data[:,0].reshape(-1),self.pb_data[:,1].reshape(-1))
        plt.scatter(self.hk_data[:,0].reshape(-1),self.hk_data[:,1][0].reshape(-1))
        plt.show()
        pass
    def let_kmeans(self):
        hk_max = 0
        hk_clusters = 0
        pb_max = 0
        pb_clusters = 0
        hk_best_model = None
        pb_best_model = None
        hk_ch_data = []
        pb_ch_data = []
        for i in range(2,15):
            print("%d times training"%i)
            hk_kmeans = KMeans(n_clusters=i,random_state=0).fit(self.hk_pca_data)
            pb_kmeans = KMeans(n_clusters=i,random_state=0).fit(self.pb_pca_data)
            hk_train_labels = hk_kmeans.labels_
            pb_train_labels = pb_kmeans.labels_
            
            # print((self.hk_data))
            # print((self.pb_data))
            hk_CH_score = calinski_harabasz_score(self.hk_pca_data, hk_train_labels)
            pb_CH_score = calinski_harabasz_score(self.pb_pca_data, pb_train_labels)
            pb_ch_data.append(pb_CH_score)
            hk_ch_data.append(hk_CH_score)
            if (hk_CH_score > hk_max):
                hk_max = hk_CH_score
                hk_clusters = i
                hk_best_model = hk_kmeans
            if(pb_CH_score > pb_max):
                pb_max = pb_CH_score
                pb_best_model = pb_kmeans
                pb_clusters = i
        # cluster_ch 图
        
        plt.title("clusters_ch")
        plt.plot(range(2,15),hk_ch_data)
        plt.plot(range(2,15),pb_ch_data)
        plt.legend(['hk_ch','pb_ch'])
        plt.show()
        
        
        writer = pd.ExcelWriter('clusters_sheet_1.xlsx')
        counter = 0
        for sy in  range(4):
            counter += 1
            if sy == 0:
                data = pd.DataFrame(self.hk_data)
                data.columns = self.loader.columns_names
            elif sy == 1:
                data = pd.DataFrame(self.pb_data)
                data.columns = self.loader.columns_names
            elif sy == 2:
                data = pd.DataFrame(hk_best_model.cluster_centers_)
                #data.columns = self.loader.columns_names
            else:
                data = pd.DataFrame(pb_best_model.cluster_centers_)
                #data.columns = self.loader.columns_names
            data.to_excel(writer,'sheet'+str(counter))
        writer.save()
        writer.close()
        print('hk_chscore: \n',hk_max,"\nclusters: %d"%hk_clusters)
        #print('hk_clusters:\n',hk_kmeans.cluster_centers_)
        # print(self.fy_part_true_kind)
        print('pb_chscore: \n',pb_max,"\nclusters: %d"%pb_clusters)
        #print('pb_clusters:\n',pb_kmeans.cluster_centers_)
        # print(self.no_fy_part_true_kind)
        return [hk_train_labels,pb_train_labels]

# DL = dataloader()
# DL.relative_judge()
'''

'''
# r1 = relic(1)
# print(r1.data)

# kmeans_solver = kmeans_solver_1()
# print(len(kmeans_solver.fy_part_true_kind)+len(kmeans_solver.no_fy_part_true_kind))


dl = dataloader()
# dl.enrich_dataset()

# km =  kmeans_solver_2()
# km.let_kmeans()

# km.plot_show()
