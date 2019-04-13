import pandas as pd
import os
import numpy as np

TOOL_SIGNAL_DATA_ROOT_DIR = "./tool_signal_data_equal/"

CACHE_ROOT_DIR = "./cache"

SIGNAL_CACHE_PATH = os.path.join(CACHE_ROOT_DIR,"SIGNAL_CACHE")

ALL_DATA_PATH = "./experiment_data.xlsx"

SURFACE_ROUGHNESS_LABEL_PATH = "./surface_roughness_labels"

MAX_TOOL_NUM = 18

class dataSet(object):

    SPINDLE_SPEED = np.array([3000,3000,3000,3000,3000,3000,4000,4000,4000,4000,4000,4000,2000,2000,2000,2000,2000,2000,])

    FEED_RATE = np.array([200,200,200,250,250,250,150,150,150,250,250,250,150,150,150,200,200,200,])

    CUT_DEPTH = np.array([0.5,0.5,0.5,1.5,1.5,1.5,1.5,1.5,1.5,0.5,0.5,0.5,0.5,0.5,0.5,1.5,1.5,1.5,])

    def __init__(self):
        pass

    def fast_get_raw_data(self):
        signal_raw_data = None
        if os.path.exists(SIGNAL_CACHE_PATH+".npy"):
            signal_raw_data = np.load(SIGNAL_CACHE_PATH+".npy")
        else:
            signal_raw_data = self.get_raw_data()
            if not os.path.exists(CACHE_ROOT_DIR):
                os.mkdir(CACHE_ROOT_DIR)
            np.save(SIGNAL_CACHE_PATH,signal_raw_data)
        return signal_raw_data

    def get_raw_data(self):
        all_signal_data = np.array([])
        for cutIdx in range(1,19):
            TOOL_SIGNAL_DIR_PATH = os.path.join(TOOL_SIGNAL_DATA_ROOT_DIR,"cut_%d"%(cutIdx))
            for machIdx in range(1,100):
                MACHINING_SIGNAL_PATH = os.path.join(TOOL_SIGNAL_DIR_PATH,"%d.csv"%(machIdx))
                if not os.path.exists(MACHINING_SIGNAL_PATH):
                    break
                else:

                    each_signal_data = pd.read_csv(MACHINING_SIGNAL_PATH)
                    # print("HEADER",each_signal_data.columns)
                    if "rms" in each_signal_data:
                        if "Fx" in each_signal_data:
                            each_signal_data = each_signal_data[[ 'Fx', 'Fy', 'Fz', 'ax', 'ay', 'az', 'rms', 'rms_filter']]
                        elif "fx" in each_signal_data:
                            each_signal_data = each_signal_data[['fx', 'fy', 'fz', 'ax', 'ay', 'az', 'rms', 'rms_filter']]
                        else:

                            raise BaseException("NO Force in X direction found")
                    elif "AE" in each_signal_data:
                        each_signal_data = each_signal_data[['fx', 'fy', 'fz', 'ax', 'ay', 'az','AE', 'filter',]]
                    else:
                        print("! HEADER", each_signal_data.columns)
                        raise BaseException("No MATCHED COLUMNS")

                    print(cutIdx, machIdx,each_signal_data.shape)
                    if cutIdx == 1 and machIdx == 1:
                        all_signal_data = np.array(each_signal_data)[np.newaxis,:,:]
                    else:
                        each_signal_data = np.array(each_signal_data)[np.newaxis,:,:]
                        all_signal_data = np.append(all_signal_data,each_signal_data,axis=0)
        print(all_signal_data.shape)
        return all_signal_data



    def interp_tensor(self,x_a:int,x_b:int,y_a,y_b,inter_x:int):
        return (y_b - y_a) / (x_b - x_a) * (inter_x - x_a) + y_a

    def search_and_interp_value(self,surface_roughness_data:pd.DataFrame,idx:int):
        id_number_pd_list = surface_roughness_data.iloc[5:,0]
        id_number_value_list = id_number_pd_list.values

        if idx in id_number_value_list:
            column = id_number_pd_list[id_number_value_list == idx].index.tolist()[0]
            print(idx,column,np.array(surface_roughness_data.iloc[column,1:5]))
            return np.array(surface_roughness_data.iloc[column,1:5])
        else:
            left_column = id_number_pd_list[id_number_value_list < idx].index.tolist()[-1]
            right_column = id_number_pd_list[id_number_value_list > idx].index.tolist()[0]
            left_roughness = np.array(surface_roughness_data.iloc[left_column,1:5])
            right_roughness = np.array(surface_roughness_data.iloc[right_column, 1:5])
            print(idx,left_column,right_column)



    def get_surface_roughness_label(self):
        all_roughness_data = None
        for cutIdx in range(1,19):
            REAL_TOOL_IDX = cutIdx+2
            each_surface_roughness_path = os.path.join(SURFACE_ROUGHNESS_LABEL_PATH,"%d号刀.xlsx"%(REAL_TOOL_IDX))
            each_surface_roughness_np_labels = pd.read_excel(each_surface_roughness_path)
            each_surface_roughness_labels = np.array(each_surface_roughness_np_labels.iloc[0:,4])
            if cutIdx == 1:
                all_roughness_data = np.array(each_surface_roughness_labels)[np.newaxis, :]
            else:
                each_roughness_data = np.array(each_surface_roughness_labels)[np.newaxis, :]
                all_roughness_data = np.append(all_roughness_data, each_roughness_data)

        print(all_roughness_data.shape)
        return all_roughness_data

    def check_data_labels(self):
        for cutIdx in range(1, 19):
            REAL_TOOL_IDX = cutIdx + 2
            each_surface_roughness_path = os.path.join(SURFACE_ROUGHNESS_LABEL_PATH, "%d号刀.xlsx" % (REAL_TOOL_IDX))
            each_surface_roughness_np_labels = pd.read_excel(each_surface_roughness_path)
            each_surface_roughness_labels = np.array(each_surface_roughness_np_labels.iloc[0:, 4])
            TOOL_SIGNAL_DIR_PATH = os.path.join(TOOL_SIGNAL_DATA_ROOT_DIR, "cut_%d" % (cutIdx))
            MAX_TOOL = 0
            for machIdx in range(1,100):
                MACHINING_SIGNAL_PATH = os.path.join(TOOL_SIGNAL_DIR_PATH,"%d.csv"%(machIdx))
                if not os.path.exists(MACHINING_SIGNAL_PATH):
                    break
                else:
                    MAX_TOOL = machIdx
            if MAX_TOOL != each_surface_roughness_labels.shape[0]:
                print("!",cutIdx,each_surface_roughness_labels.shape[0],MAX_TOOL)
            else:
                print("#",cutIdx, each_surface_roughness_labels.shape[0], MAX_TOOL)

    def get_run_number_data(self):
        run_label = []
        for cutIdx in range(1, 19):
            REAL_TOOL_IDX = cutIdx + 2
            each_surface_roughness_path = os.path.join(SURFACE_ROUGHNESS_LABEL_PATH, "%d号刀.xlsx" % (REAL_TOOL_IDX))
            each_surface_roughness_np_labels = pd.read_excel(each_surface_roughness_path)
            each_surface_roughness_labels = np.array(each_surface_roughness_np_labels.iloc[0:, 4])

            MAX_TOOL_NUM = each_surface_roughness_labels.shape[0]
            run_label.extend([i+1 for i in range(MAX_TOOL_NUM)])
        return np.array(run_label)

    def get_condition_number_data(self):
        run_label = []
        spindle_labels = []
        feed_rates = []
        cut_depths = []
        for cutIdx in range(1, 19):
            REAL_TOOL_IDX = cutIdx + 2
            each_surface_roughness_path = os.path.join(SURFACE_ROUGHNESS_LABEL_PATH, "%d号刀.xlsx" % (REAL_TOOL_IDX))
            each_surface_roughness_np_labels = pd.read_excel(each_surface_roughness_path)
            each_surface_roughness_labels = np.array(each_surface_roughness_np_labels.iloc[0:, 4])

            MAX_TOOL_NUM = each_surface_roughness_labels.shape[0]
            run_label.extend([i+1 for i in range(MAX_TOOL_NUM)])
            spindle_labels.extend([self.SPINDLE_SPEED[cutIdx-1]]*MAX_TOOL_NUM)
            feed_rates.extend([self.FEED_RATE[cutIdx-1]] * MAX_TOOL_NUM)
            cut_depths.extend([self.CUT_DEPTH[cutIdx-1]] * MAX_TOOL_NUM)
        condition_data = np.zeros((451,4))
        condition_data[:,0] = np.array(run_label)
        condition_data[:, 1] = np.array(spindle_labels)
        condition_data[:, 2] = np.array(feed_rates)
        condition_data[:, 3] = np.array(cut_depths)
        return condition_data
        # return np.array(run_label),np.array(spindle_labels),np.array(feed_rates),np.array(cut_depths)

    def get_reinforced_data(self):
        label = self.get_surface_roughness_label()
        signal = self.fast_get_raw_data()
        index = np.array([i for i in range(0,10000,20)])
        # sample that
        reinforced_signal = []
        reinforced_label = []
        for sample_idx in range(signal.shape[0]):
            for i in range(0,20):
                print(signal[sample_idx].shape)
                cur_signal = signal[sample_idx]
                reinforced_signal.append(cur_signal[index+i])
                reinforced_label.append(label[sample_idx])
        print(np.array(reinforced_label).shape,np.array(reinforced_signal).shape)
        return np.array(reinforced_signal),np.array(reinforced_label)

    def get_reinforced_condition_data(self):
        condition_data = self.get_condition_number_data()
        reinforced_data = np.zeros((9020,4))
        for sample_idx in range(condition_data.shape[0]):
            for i in range(0,20):
                reinforced_data[sample_idx*20+i,:] = condition_data[sample_idx,:]
        return reinforced_data

    def get_test_show_data(self):
        label = self.get_surface_roughness_label()
        signal = self.fast_get_raw_data()
        index = np.array([i for i in range(0, 10000, 20)])
        reinforced_signal = []
        reinforced_label = []
        for sample_idx in range(signal.shape[0]):

            cur_signal = signal[sample_idx]
            reinforced_signal.append(cur_signal[index])
            reinforced_label.append(label[sample_idx])
        print(np.array(reinforced_label).shape, np.array(reinforced_signal).shape)
        return np.array(reinforced_signal), np.array(reinforced_label)



if __name__ == "__main__":
    data = dataSet()
    data.fast_get_raw_data()
    # run,spindle,feed,cut = data.get_condition_number_data()
    # print(run.shape,spindle.shape,feed.shape,cut.shape)
    signal,label = data.get_reinforced_data()
    # print(data.get_run_number_data().shape)
    print(signal.max(axis=0).max(axis=0))
    print(signal.min(axis=0).min(axis=0))
    print(data.get_condition_number_data().shape)
    print(data.get_reinforced_condition_data().shape)
    # print(data.SPINDLE_SPEED.shape,data.FEED_RATE.shape,data.CUT_DEPTH.shape)

