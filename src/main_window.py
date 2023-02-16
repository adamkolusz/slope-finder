import tkinter
import customtkinter
import os
from PIL import Image, ImageTk
import angle_detector
import cv2
import json


class WidgetSettings:
    def __init__(
        self,
        row=0,
        column=0,
        padx=10,
        pady=10,
        ipadx=5,
        ipady=5,
        width=100,
        height=25,
        fill="none",
        sticky="ne",
        expand=False,
        side="top",
    ):
        self.row = row
        self.column = column
        self.padx = padx
        self.pady = pady
        self.ipadx = ipadx
        self.ipady = ipady
        self.width = width
        self.height = height
        self.fill = fill
        self.sticky = sticky
        self.expand = expand
        self.side = side


class GridSettings:
    def __init__(
        self,
        column=0,
        row=0,
        columnspan=1,
        rowspan=1,
        in_=None,
        width=50,
        height=25,
        ipadx=5,
        ipady=5,
        padx=5,
        pady=5,
        sticky="nsew",
        corner_radius=5,
    ):
        self.row = row
        self.column = column
        self.columnspan = columnspan
        self.rowspan = rowspan
        self.in_ = in_
        self.width = width
        self.height = height
        self.padx = padx
        self.pady = pady
        self.ipadx = ipadx
        self.ipady = ipady
        self.sticky = sticky
        self.corner_radius = corner_radius
        self.image_path = ""


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Super Apka")
        self.geometry("1600x900")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        frameStyle = GridSettings(0, 0, 0, 0, None, 50, 25, 0, 0, 0, 0, "nswe")
        homeStyle = GridSettings(0, 0, 0, 0, None, 50, 25, 5, 5, 10, 10, "nswe")
        tabStyle = GridSettings(0, 0, 1, 1, None, 50, 25, 0, 0, 10, 10, "nswe")
        labelStyle = GridSettings(0, 0, 0, 0, None, 50, 25, 0, 0, 0, 0, "nsew")
        sliderStyle = GridSettings(0, 0, 0, 0, None, 50, 25, 0, 0, 0, 00, "nsw")
        entryStyle = GridSettings(0, 0, 0, 0, None, 50, 25, 0, 0, 0, 0, "nsw")

        # LAYOUT
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(
            row=frameStyle.row,
            column=frameStyle.column,
            padx=frameStyle.padx,
            pady=frameStyle.pady,
            sticky=frameStyle.sticky,
        )

        self.navigation_frame.grid_rowconfigure(0, weight=1)
        self.navigation_frame.grid_columnconfigure(0, weight=1)

        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.home_frame.grid(row=0, column=1)

        self.home_frame.grid_rowconfigure(0, weight=1)
        self.home_frame.grid_columnconfigure(0, weight=1)
        self.tab_name_lst = ["Options", "Filter", "Position"]

        self.widgets = {}

        with open("data.json", "r") as fp:
            self.widgets = json.load(fp)
        print(self.widgets)

        self.tst_lst = []
        for i, name in enumerate(self.widgets):
            print(self.widgets[name]["int"])
            if self.widgets[name]["int"] == 1:
                self.tst_lst.append(tkinter.IntVar())
            else:
                self.tst_lst.append(tkinter.DoubleVar())

        print(self.tst_lst)
        with open("data.json", "w") as fp:
            json.dump(self.widgets, fp, sort_keys=True, indent=4)

        self.tabview = customtkinter.CTkTabview(self.navigation_frame)
        self.tabview.grid(
            row=tabStyle.row,
            column=tabStyle.column,
            rowspan=tabStyle.rowspan,
            ipadx=tabStyle.ipadx,
            ipady=tabStyle.ipady,
            padx=tabStyle.padx,
            pady=tabStyle.pady,
            sticky=tabStyle.sticky,
        )

        # self.tabview.grid_rowconfigure(1, weight=1)
        # self.tabview.grid_columnconfigure(1, weight=1)
        for name in self.tab_name_lst:
            self.tabview.add(name)
        self.tabview.set(self.tab_name_lst[1])

        for i, name in enumerate(self.widgets):
            self.label = customtkinter.CTkLabel(
                master=self.tabview.tab(self.tab_name_lst[self.widgets[name]["tab"]]),
                text=name,
            )
            self.label.grid(
                row=labelStyle.row + i * 2,
                column=labelStyle.column,
                padx=labelStyle.padx,
                pady=labelStyle.pady,
                sticky=labelStyle.sticky,
                columnspan=2,
            )

            self.sliderpack = customtkinter.CTkSlider(
                master=self.tabview.tab(self.tab_name_lst[self.widgets[name]["tab"]]),
                from_=self.widgets[name]["min"],
                to=self.widgets[name]["max"],
                number_of_steps=self.widgets[name]["step"],
                command=self.refresh_image,
                variable=self.tst_lst[i],
            )

            self.sliderpack.grid(
                row=sliderStyle.row + 1 + i * 2, column=sliderStyle.column
            )

            self.entry = customtkinter.CTkEntry(
                master=self.tabview.tab(self.tab_name_lst[self.widgets[name]["tab"]]),
                placeholder_text="CTkEntry",
                textvariable=self.tst_lst[i],
                width=entryStyle.width,
                height=entryStyle.height,
            )
            # self.entry.bind("<Return>", command=self.get_text)

            self.entry.grid(
                row=sliderStyle.row + 1 + i * 2,
                column=sliderStyle.column + 1,
                padx=entryStyle.padx,
                pady=entryStyle.pady,
            )

        self.image_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "C:/Users/adamk/Projects/cone_detection/img/fluent/Results",
        )

        self.logo_image = customtkinter.CTkImage(
            Image.open(os.path.join(self.image_path, "e6_fluent_animation0.png")),
            size=(100, 100),
        )

        self.img_label = customtkinter.CTkLabel(self.home_frame, image=self.logo_image)
        self.img_label.grid(row=0, column=1, padx=20, pady=10)
        self.CVapp = angle_detector.AngleDetector()
        self.CVapp.load_config()
        # CVapp.main()

    def update_params(self):
        self.CVapp.top_boundary = self.widgets["top_boundary"]["value"]

    def refresh_image(self, val):
        self.img = cv2.imread(
            "C:/Users/adamk/Projects/cone_detection/img/fluent/Results/e6_fluent_animation39.png"
        )
        self.update_params()
        _, colour_image, _ = self.CVapp.calculate_lines(self.img)
        colour_image = cv2.cvtColor(colour_image, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(colour_image, mode="RGB")
        imgtk = ImageTk.PhotoImage(image=im)
        self.img_label.configure(image=imgtk)

    # def get_text(self, event):
    #     # self.sliderpack.set(int(self.v.get()))
    #     # self.label.configure(text=int(self.v.get()))
    #     self.entry.select_clear()
    #     # self.entry.delete(0, "end")

    #     print(self.v)

    # def slider_event_pack(self, value):
    #     # self.numberboxpack.insert("0.0", round(value, 5))
    #     # self.label.configure(text=int(self.v.get()))
    #     # self.entry.delete(0, "end")
    #     self.entry.select_clear()
    #     print(value)

    # def slider_event_2(self, value):
    #     self.textbox4.insert("0.0", round(value, 5))
    #     print(value)


if __name__ == "__main__":
    app = App()
    app.mainloop()
