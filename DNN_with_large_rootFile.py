from training_accessories_march_23 import *
import math
import pickle

#root_file = "/media/miguel/Elements/Data_hcali/Data1/Jan_2023_log_space_Files/log_uniform_e-_17deg_jan26_23.root"
root_file="/media/miguel/Elements/Data_hcali/Data1/log10_Uniform_03-23/log10_pi+_17deg_2.2M.root"
new_dir="log10pi+_lr5_maxDim4"
path="/home/bishnu/EIC/output_reg_dnn_straw/"

batch_size = 1000
N_Events=1000
Events_for_mean=4000
detector_name="hcal"
log10=False
learning_rate = 5e-5
dropout_rate = 0.05
N_Epochs =50
loss = 'mae'
write_Y=True
## how many features you want to input in your training 
## 2 EZ, 3=> EZX , 4=> EXYZ. , 5=> EXYZT
max_dim=4
chunk_size=1000


######## NOTHING NEEDS TO BE CHANGED BELOW THIS
ur_file = ur.open(root_file)
ur_tree = ur_file['events']
num_entries=ur_tree.numentries
print("Total Entries ",num_entries)

### GET MEAN AND STANDARD DEVIATION FROM FIRST FEW SAMPLES
hit_e, PosRecoX ,PosRecoY ,PosRecoZ ,time, gen_energy=read_root_file(root_file, N_Events, detector_name)
max_length = max(len(seq) for seq in hit_e)

if log10==True:
    hit_e[hit_e<0.000005]=0.000005 # to avoid the log10(0)
    hit_e=get_log10_hitE(hit_e)
    gen_energy=np.log10(gen_energy)

input_hit_info=[hit_e, PosRecoZ ,PosRecoX ,PosRecoY ,time]


## Mean and standard deviation for input variable
means, stds=get_mean_std_input(input_hit_info, Events_for_mean)

## Mean and standard deviation of target variable 
mean_target, std_target=get_mean_std_target(gen_energy,Events_for_mean)

#print(mean_target)

output_path=f"{path}{new_dir}"
ifExists=os.path.exists(output_path)
if ifExists==True:
    shutil.rmtree(output_path)
    os.mkdir(output_path)
if ifExists==False:
    os.mkdir(output_path)
    
#print(output_path)

val_fraction=0.3
train_fraction=0.5
test_fraction=1-val_fraction-train_fraction

train_start=0
train_end=int(num_entries*train_fraction)

val_start=train_end
val_end=train_end + int(num_entries*val_fraction)

test_start=val_end
test_end=num_entries


#data_generator(file_path, detector, entry_start, stop, batch_size=batch_size)

train_dataset = tf.data.Dataset.from_generator(
    generator=data_generator,
    args=[root_file, detector_name.encode('utf-8'), train_start, train_end, means, stds, mean_target, std_target,max_dim, max_length, batch_size, chunk_size, log10],  
    output_types=(tf.float64, tf.float64),
    output_shapes=([ None, None,max_dim], [None])
)

validation_dataset = tf.data.Dataset.from_generator(
    generator=data_generator,
    args=[root_file, detector_name.encode('utf-8'), val_start,val_end, means, stds, mean_target, std_target,max_dim, max_length,batch_size,chunk_size,log10],  
    output_types=(tf.float64, tf.float64),
    output_shapes=([None,None, max_dim], [None])
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(max_length,max_dim)), #// for more than one input
    #tf.keras.layers.Input(shape=(num_pixels_2)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1,activation="linear")## output layer
])


# Compile the model
model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

history_logger=tf.keras.callbacks.CSVLogger(output_path+"/log.csv", separator=",", append=True)
callbacks=[history_logger]

history=model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=N_Epochs,
    batch_size=batch_size,
    callbacks=[callbacks]
    
)


test_dataset = tf.data.Dataset.from_generator(
    generator=data_generator,
    args=[root_file, detector_name.encode('utf-8'), test_start,test_end, means, stds, mean_target, std_target,max_dim, max_length, batch_size,chunk_size, log10],  
    output_types=(tf.float64, tf.float64),
    output_shapes=([None,None, max_dim], [None])
)

model.save('%s/dnn_model.h5'%(output_path))

X_test = []
Y_test = []
for x, y in test_dataset.as_numpy_iterator():
    X_test.append(x)
    y_temp=(y*std_target) + mean_target
    if log10==True:
        y=10**y_temp
    else:
        y=y_temp
        
    Y_test.append(y)


if write_Y:
    write_Y_test(batch_size, Y_test, output_path)

mypreds = []
for x_test_batch in X_test:
    preds_batch = model.predict_on_batch(x_test_batch)
    mypreds.append(preds_batch)

mypreds = np.concatenate(mypreds, axis=0)
mypreds_actual=mypreds*std_target + mean_target
np.save("%s/R_prediction.npy"%(output_path), mypreds_actual)



mean_std=np.array([mean_target, std_target])
np.save("%s/mean_std_target.npy"%(output_path),mean_std)
#np.save("%s/R_prediction.npy"%(output_path), mypreds)
