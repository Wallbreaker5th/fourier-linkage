构造使用傅里叶级数拟合曲线的连杆系统。需要安装 [manimgl](github.com/3b1b/manim)。

`fold_ft.py` 为程序主体，`ft.py` 可以预览图案，`tri.svg` 为希望绘制的图案。

你可能需要将 manimgl 源码中 `mobject/mobject.py` 文件中 `Mobject.assemble_family` 函数改为下面这样以让程序以合理的速度运行：
```python
    def assemble_family(self):
        sub_families = (sm.get_family() for sm in self.submobjects)
        self.family = [self, *it.chain(*sub_families)]
        self.family = list(set(self.family)) # add this line
        self.refresh_has_updater_status()
        self.refresh_bounding_box()
        for parent in self.parents:
            parent.assemble_family()
        return self
```
