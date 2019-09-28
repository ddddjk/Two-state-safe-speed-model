import numpy as np
import matplotlib.pyplot as plt
import time

############################ 函数定义 #########################################
# 根据只包含车路位置和速度的一维数组转换成包含速度和位置的二维数组
def transfer(vehicle_number,road_length,vehicle_position,vehicle_velocity):
    cells_data = np.zeros((2,road_length),dtype = int)
    for i in range(vehicle_number):
        index = int(vehicle_position[i])
        cells_data[0,index] = 1
        cells_data[1,index] = vehicle_velocity[i]
    return cells_data

def gap(x,road_length,vehicle_length):
    temp = x[0]
    leading = np.delete(x,0,axis=0)
    leading = np.append(leading,temp+road_length)
    diff = leading - x - vehicle_length * np.ones_like(leading)
    if diff.any()>0:
        return diff
    else:
        raise ValueError('There is a negative value in gap')

def tail_to_lead(v):
    temp = v[0]
    res = np.delete(v,0,axis=0)
    res = np.append(res,temp)
    return res

def safe_speed(b_max,v_lead,d):
    v_safe = np.sqrt(b_max*b_max + v_lead*v_lead + 2*b_max*d) - b_max
    v_safe = np.floor(v_safe)
    return v_safe

def posi_init(vehicle_number,distri_form,vehicle_length=15,road_length=7500):
    posi = np.zeros(0,dtype=int)
    if distri_form == "homogenous":
        init_space = int(road_length/vehicle_number)
        posi = np.append(posi,vehicle_length)
        for i in range(1,vehicle_number):
            posi = np.append(posi,vehicle_length + i*init_space)
        return posi
    elif distri_form == "megajam":
        for nn in range(vehicle_number):
            posi = np.append(posi,1+vehicle_length*nn)
        return posi
    else:
        raise Exception("Please input a valid distributed form")

############################ 参数定义 #########################################
road_length = 7500

vehicle_length = 15
v_max = 60 #maximum speed
T = 1.8 # safe time gap
pa = 0.85 #随机慢化概率
pb = 0.52
pc = 0.1
a = 1
b_max = 7
b_defense = 2
g_safety = 20
v_c = 30
alpha = 10

steps = 12000 #仿真步长
densities = range(27,28)
v_avg = np.zeros(0,dtype=float)

time_start = time.time()

for density in densities:
    ######################## 声明过程数据 #####################################
    velocity_avg = 0
    vehicle_number = int(road_length/2/1000*density)

    velocities = np.zeros((0,vehicle_number),dtype=int) #记录速度过程信息
    velocities_vehi = np.zeros((0,vehicle_number),dtype=int) #以车辆顺序记录速度过程信息
    positions = np.zeros((0,vehicle_number),dtype=int) #记录位置过程信息
    positions_vehi = np.zeros((0,vehicle_number),dtype=int) #以车辆顺序记录位置过程信息
    
    vehicle_velocity = np.zeros(0,dtype=int) #存储车辆速度
    vehicle_position = np.zeros(0,dtype=int) #存储车辆的位置
    
    ######################## 初始化 ##########################################
    vehicle_position = posi_init(vehicle_number,"megajam")
    vehicle_velocity = np.ones(vehicle_number)*v_max #赋予车辆初始速度为 Vmax
    
    #将初始化数据放入velocities和positions中
    velocities = np.append(velocities,[vehicle_velocity],axis=0)    
    positions = np.append(positions,[vehicle_position],axis=0)
    velocities_vehi = np.append(velocities_vehi,[vehicle_velocity],axis=0)    
    positions_vehi = np.append(positions_vehi,[vehicle_position],axis=0)
    
    transfet_counts = 0 #记录首车变尾车的次数
    
    ##################### 迭代 ###############################################
    for i in range(steps):
        p = np.zeros(0,dtype=int) #存储车辆随机减速的概率
        b_rand = np.zeros(0,dtype=int) #存储车辆随机减速的减速度大小
        
        #第零步：确定p和b_rand
        current_gap = gap(vehicle_position,road_length,vehicle_length)
        leader_gap = tail_to_lead(current_gap)
        v_leader = tail_to_lead(vehicle_velocity)
        v_anti = np.min(np.vstack((leader_gap,v_leader+a,\
                                   v_max*np.ones_like(v_leader))),axis=0)
        d_anti = current_gap + np.max(np.vstack((v_anti-g_safety,np.zeros_like(v_anti))),axis=0)
        v_safe = safe_speed(b_max,v_leader,current_gap)
        
        for j in range(vehicle_number):
            if vehicle_velocity[j]==0:
                p = np.append(p,pb)
            elif vehicle_velocity[j] <= (d_anti[j]/T):
                p = np.append(p,pc)
            else:
                p_defense = pc + pa/(1+pow(np.e,alpha*(v_c-vehicle_velocity[j])))
                p = np.append(p,p_defense)
        for k in range(vehicle_number):
            if vehicle_velocity[k] < b_defense + np.floor(d_anti[k]/T):
                b_rand = np.append(b_rand,a)
            else:
                b_rand = np.append(b_rand,b_defense)
            
        #第一步：加速
        vehicle_velocity = np.min(np.vstack((vehicle_velocity+a,\
                                             v_max*np.ones_like(vehicle_velocity),d_anti,v_safe)),axis=0)
        
        #第二步：随机慢化
        prob = np.random.random(vehicle_number)
        v_above = np.multiply(vehicle_velocity,np.array(prob>=p,dtype=int))
        v_below = np.multiply(vehicle_velocity,np.array(prob<p,dtype=int))
        v_slow = np.amax(np.vstack((v_below-b_rand,np.zeros(vehicle_number))),axis=0)
        vehicle_velocity = v_slow + v_above
        
        #第三步：位置更新
        vehicle_position = vehicle_position + vehicle_velocity
        if vehicle_position[-1] >= road_length:
            transfet_counts += 1 
            
            temp_pos = vehicle_position[-1] - road_length
            vehicle_position = np.delete(vehicle_position,-1,0)
            vehicle_position = np.insert(vehicle_position,0,temp_pos,0)
            temp_vel = vehicle_velocity[-1]
            vehicle_velocity = np.delete(vehicle_velocity,-1,0)
            vehicle_velocity = np.insert(vehicle_velocity,0,temp_vel,0)

            posi = transfet_counts%vehicle_number
            vehicle_velocity_new = np.concatenate((vehicle_velocity[posi:],vehicle_velocity[:posi]))
            vehicle_position_new = np.concatenate((vehicle_position[posi:],vehicle_position[:posi]))
            velocities_vehi = np.append(velocities_vehi,[vehicle_velocity_new],axis=0)
            positions_vehi = np.append(positions_vehi,[vehicle_position_new],axis=0)                        
        else:
            posi = transfet_counts%vehicle_number
            vehicle_velocity_new = np.concatenate((vehicle_velocity[posi:],vehicle_velocity[:posi]))
            vehicle_position_new = np.concatenate((vehicle_position[posi:],vehicle_position[:posi]))
            velocities_vehi = np.append(velocities_vehi,[vehicle_velocity_new],axis=0)
            positions_vehi = np.append(positions_vehi,[vehicle_position_new],axis=0) 
            
#######此时velocities 和 positions 中存放数据是车辆对应的数组位置在变的数据形式
        velocities = np.append(velocities,[vehicle_velocity],axis=0)
        positions = np.append(positions,[vehicle_position],axis=0)
######################## 计算密度、流量 ################################
#    for i in range(4000,5000):
#        velocity_avg += np.sum(velocities[i,:])/vehicle_number
#    velocity_avg = velocity_avg/1000/2*3.6
#    v_avg = np.append(v_avg,velocity_avg)
#    print('The density of %3d is done!!!' % density)
        
time_end = time.time()
time_cost = time_end - time_start
print('Totally used %d s' %time_cost)

############################# 画基本图 ################################### 
#plt.plot(densities,np.multiply(densities,v_avg),'*')
#plt.title('Flow-Density Diagram')
#plt.xlabel('Denstity')
#plt.ylabel('Flow')


hori_axis = np.tile(np.arange(10000,11000,dtype=int),np.arange(0,vehicle_number,5).size).reshape(np.arange(0,vehicle_number,5).size,1000)
hori_axis = hori_axis.T
fig, ax = plt.subplots()
points = ax.scatter(hori_axis, positions_vehi[10000:11000,0::5], c=velocities_vehi[10000:11000,0::5],marker=',',s=1,vmin=0,vmax=60)
plt.colorbar(points)
plt.show()