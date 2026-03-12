# 万有引力的模拟实验

## 1.编辑物理系统
  
    import taichi as ti
    from .config import *
    
    # 1. 数据结构定义：在显存中开辟空间
    pos = ti.Vector.field(2, dtype=float, shape=NUM_PARTICLES)
    vel = ti.Vector.field(2, dtype=float, shape=NUM_PARTICLES)
    
    @ti.kernel
    def init_particles():
        """初始化每一个粒子的随机坐标"""
        for i in range(NUM_PARTICLES):
            pos[i] = [ti.random(), ti.random()]
            vel[i] = [0.0, 0.0]
    
    @ti.kernel
    def update_particles(mouse_x: float, mouse_y: float):
        """物理更新：由 GPU 并行执行"""
        for i in range(NUM_PARTICLES):
            # 计算方向与距离
            mouse_pos = ti.Vector([mouse_x, mouse_y])
            dir = mouse_pos - pos[i]
            dist = dir.norm()
            
            # 施加引力与阻力
            if dist > 0.05:
                vel[i] += dir.normalized() * GRAVITY_STRENGTH
                
            vel[i] *= DRAG_COEF  
            pos[i] += vel[i]
             # 边框碰撞检测
            for j in ti.static(range(2)):
                if pos[i][j] < 0:
                    pos[i][j] = 0.0
                    vel[i][j] *= BOUNCE_COEF
                elif pos[i][j] > 1:
                    pos[i][j] = 1.0
                    vel[i][j] *= BOUNCE_COEF
## 2.控制物理参数
   
    --- 物理系统参数 ---
    NUM_PARTICLES = 10000      # 粒子总数 (卡顿请调小此数值，如        2000)
    GRAVITY_STRENGTH = 0.001   # 鼠标引力强度
    DRAG_COEF = 0.98           # 空气阻力系数
    BOUNCE_COEF = -0.8         # 边界反弹能量损耗# --- 渲染系统参      数 ---
    WINDOW_RES = (800, 600)    # 窗口分辨率
    PARTICLE_RADIUS = 1.5      # 粒子绘制半径
    PARTICLE_COLOR = 0x00BFFF  # 粒子颜色 (天蓝色)
    
## 3.编写主程序
    # src/Work0/main.py
    import taichi as ti
    
    # 注意：初始化必须在最前面执行，接管底层 GPU
    ti.init(arch=ti.gpu)
    
    # 导入我们自己写的模块
    from .config import WINDOW_RES, PARTICLE_COLOR, PARTICLE_RADIUS
    from .physics import init_particles, update_particles, pos
    
    def run():
        print("正在编译 GPU 内核，请稍候...")
        init_particles()
        
        gui = ti.GUI("Experiment 0: Taichi Gravity Swarm", res=WINDOW_RES)
        print("编译完成！请在弹出的窗口中移动鼠标。")
        
        # 渲染主循环
        while gui.running:
            mouse_x, mouse_y = gui.get_cursor_pos()
            
            # 驱动 GPU 进行物理计算
            update_particles(mouse_x, mouse_y)
            
            # 读取显存数据并绘制
            gui.circles(pos.to_numpy(), color=PARTICLE_COLOR, radius=PARTICLE_RADIUS)
            gui.show()
    
    if __name__ == "__main__":
        run()
## 4.最终效果展示




<!-- 仓库内 GIF，控制宽度为 600px -->
<img src="https://raw.githubusercontent.com/Changqinkai/myMAF/main/2026-03-12195438-ezgif.com-video-to-gif-converter.gif" alt="功能演示" width="600">


