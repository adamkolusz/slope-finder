import tkinter
import customtkinter
from tkinter.messagebox import showinfo
import os
from PIL import Image, ImageTk
import angle_detector
import cv2
import json
import numpy as np
import time


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


# class MyFrame(customtkinter.CTkScrollableFrame):
#     def __init__(self, master, **kwargs):
#         super().__init__(master, **kwargs)

#         # add widgets onto the frame...
#         self.label = customtkinter.CTkLabel(self)
#         self.label.grid(row=0, column=0, padx=20)


class ResizingCanvas(customtkinter.CTkCanvas):
    def __init__(self, parent, **kwargs):
        customtkinter.CTkCanvas.__init__(self, parent, **kwargs)
        self.bind("<Configure>", self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()

    def on_resize(self, event):
        # determine the ratio of old width/height to new width/height
        wscale = float(event.width) / self.width
        hscale = float(event.height) / self.height
        self.width = event.width
        self.height = event.height
        # resize the canvas
        self.config(width=self.width, height=self.height)
        # rescale all the objects tagged with the "all" tag
        self.scale("all", 0, 0, wscale, hscale)


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.init = 1
        self.title("Slope Finder")
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
        buttonStyle = GridSettings(0, 0, 2, 0, None, 50, 25, 5, 5, 5, 5, "ns")
        actionButtonStyle = GridSettings(0, 0, 0, 0, None, 25, 25, 0, 0, 0, 0, "nsw")

        # LAYOUT
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(
            row=frameStyle.row,
            column=frameStyle.column,
            padx=frameStyle.padx,
            pady=frameStyle.pady,
            sticky=frameStyle.sticky,
        )

        # self.navigation_frame.grid_rowconfigure(0, weight=1)
        # self.navigation_frame.grid_columnconfigure(0, weight=1)

        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.button_frame = customtkinter.CTkFrame(
            self.navigation_frame, corner_radius=0, height=120
        )
        # self.home_frame.grid(row=0, column=1, sticky="nw")

        # self.home_frame.grid_rowconfigure(0, weight=1)
        # self.home_frame.grid_columnconfigure(0, weight=1)
        self.navigation_frame.pack(side="left", fill="both", expand=False)
        self.home_frame.pack(side="left", fill="both", expand=True)
        self.button_frame.pack(side="bottom", fill="y", expand=False)
        self.tab_name_lst = ["Options", "Filter", "Position"]
        self.video_mode = True
        self.frame_1 = np.array([])
        self.frame_2 = np.array([])
        self.widgets = {}

        with open("config/data.json", "r", encoding="UTF8") as fp:
            self.widgets = json.load(fp)
        # print(self.widgets)

        self.tst_lst = {}
        for i, name in enumerate(self.widgets):
            if self.widgets[name]["int"] == 1:
                self.tst_lst[name] = tkinter.IntVar()
            else:
                self.tst_lst[name] = tkinter.DoubleVar()

        # print(self.tst_lst)
        with open("config/data.json", "w") as fp:
            json.dump(self.widgets, fp, indent=4)

        self.scrollable_frame = customtkinter.CTkScrollableFrame(
            master=self.navigation_frame, width=300, height=200
        )
        # self.scrollable_frame.grid(row=0, column=0, sticky="nsew")
        self.scrollable_frame.pack(side="left", fill="both", expand=True)
        self.tabview = customtkinter.CTkTabview(self.scrollable_frame)
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

        self.labels = {}
        self.sliders = {}
        self.entries = {}
        self.buttons = {}
        self.action_buttons = {}

        self.file_path = "./img/piv/piv2.png"

        self.action_commands = [
            self.next_frame,
            self.process_video,
            self.refresh_image,
            self.refresh_image,
        ]
        action_ctr = 0
        for i, name in enumerate(self.widgets):
            if self.widgets[name]["type"] == "slider":
                self.labels[name] = customtkinter.CTkLabel(
                    master=self.tabview.tab(
                        self.tab_name_lst[self.widgets[name]["tab"]]
                    ),
                    text=self.widgets[name]["text"],
                )

                self.labels[name].grid(
                    row=labelStyle.row + i * 2,
                    column=labelStyle.column,
                    padx=labelStyle.padx,
                    pady=labelStyle.pady,
                    sticky=labelStyle.sticky,
                    columnspan=2,
                )

                self.sliders[name] = customtkinter.CTkSlider(
                    master=self.tabview.tab(
                        self.tab_name_lst[self.widgets[name]["tab"]]
                    ),
                    from_=self.widgets[name]["min"],
                    to=self.widgets[name]["max"],
                    number_of_steps=self.widgets[name]["step"],
                    command=self.refresh_image,
                    variable=self.tst_lst[name],
                )

                self.sliders[name].grid(
                    row=sliderStyle.row + 1 + i * 2, column=sliderStyle.column
                )

                self.entries[name] = customtkinter.CTkEntry(
                    master=self.tabview.tab(
                        self.tab_name_lst[self.widgets[name]["tab"]]
                    ),
                    placeholder_text="CTkEntry",
                    textvariable=self.tst_lst[name],
                    width=entryStyle.width,
                    height=entryStyle.height,
                )

                # self.entry.bind("<Return>", command=self.get_text)

                self.entries[name].grid(
                    row=sliderStyle.row + 1 + i * 2,
                    column=sliderStyle.column + 1,
                    padx=entryStyle.padx,
                    pady=entryStyle.pady,
                )
            elif (
                self.widgets[name]["type"] == "button"
                and self.widgets[name]["tab"] != 10
            ):
                self.buttons[name] = customtkinter.CTkButton(
                    master=self.tabview.tab(
                        self.tab_name_lst[self.widgets[name]["tab"]]
                    ),
                    width=buttonStyle.width,
                    height=buttonStyle.height,
                    text=self.widgets[name]["text"],
                    command=self.select_file,
                )
                # Union[Callable[[], None], None]
                self.buttons[name].grid(
                    row=buttonStyle.row + i * 2,
                    column=buttonStyle.column,
                    padx=buttonStyle.padx,
                    pady=buttonStyle.pady,
                    sticky=buttonStyle.sticky,
                    columnspan=buttonStyle.columnspan,
                )
            elif (
                self.widgets[name]["type"] == "button"
                and self.widgets[name]["tab"] == 10
            ):
                self.action_buttons[name] = customtkinter.CTkButton(
                    master=self.button_frame,
                    text=str(self.widgets[name]["text"]),
                    width=actionButtonStyle.width,
                    height=actionButtonStyle.height,
                    command=self.action_commands[action_ctr],
                )
                action_ctr += 1
                # Union[Callable[[], None], None]
                # self.action_buttons[name].grid(
                #     row=actionButtonStyle.row,
                #     column=actionButtonStyle.column,
                #     padx=actionButtonStyle.padx,
                #     pady=actionButtonStyle.pady,
                #     sticky=actionButtonStyle.sticky,
                #     columnspan=1,
                # )
                self.action_buttons[name].pack(side="left", fill="both", expand=True)

        self.image_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "C:/Users/adamk/Projects/slope-finder/img/piv/",
        )

        # image = (Image.open(filename))
        self.im = ImageTk.PhotoImage(
            Image.open(os.path.join(self.image_path, "piv6.png"))
        )
        self.logo_image = customtkinter.CTkImage(
            Image.open(os.path.join(self.image_path, "piv6.png")),
            size=(100, 100),
        )

        self.mycanvas = ResizingCanvas(
            self.home_frame,
            width=850,
            height=400,
            bg="#000000",
            highlightthickness=2,
        )
        self.mycanvas.pack(fill="both", expand=True)

        # self.canvas = customtkinter.CTkCanvas(self.home_frame, width=800, height=400)
        # # self.canvas.grid(row=0, column=1, padx=0, pady=0, sticky="nsew")
        # self.canvas.pack(fill="both", expand=True)
        self.image_id = self.mycanvas.create_image(0, 0, image=self.im, anchor="nw")

        # self.img_label = customtkinter.CTkLabel(self.home_frame, image=self.logo_image)
        # # self.img_label.grid(row=0, column=1, padx=20, pady=10)
        # self.canvas.pack(fill="both", expand=True)

        # self.canvas.bind("<Configure>", self.resizer)
        # self.mycanvas.bind("<Configure>", self.refresh_image)

        self.CVapp = angle_detector.AngleDetector()
        self.CVapp.load_config()
        # CVapp.main()

    def initial_values(self):
        self.init = 0
        for i, name in enumerate(self.widgets):
            if self.widgets[name]["type"] == "slider":
                self.tst_lst[name].set(self.widgets[name]["value"])

    def save_params(self):
        for i, name in enumerate(self.widgets):
            if self.widgets[name]["type"] == "slider":
                # print(f"{self.sliders[name].get()=}")
                # print(f"{self.tst_lst[name].get()=}")
                self.widgets[name]["value"] = self.tst_lst[name].get()
        # self.widgets["bottom_boundary"]["value"] = int(self.sliders[1].get())
        # self.widgets["top_boundary"]["value"] = int(self.sliders[0].get())

    def update_params(self):
        # self.CVapp.top_boundary = self.widgets["top_boundary"]["value"]
        self.CVapp.data_json = self.widgets
        # self.CVapp.bottom_boundary = self.widgets["bottom_boundary"]["value"]
        # print("EEEEEEEEEEEEE", self.widgets)
        # self.CVapp.top_boundary = 120

    def update_frames(self):
        if self.video_mode:
            cap = cv2.VideoCapture(self.file_path)
            while cap.isOpened():
                if True:  # cv2.waitKey(0) & 0xFF == ord('q'):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    try:
                        current_frame = frame
                        angle_array, _, _ = self.calculate_lines(current_frame)
                    except:
                        print("Could not obtain line")
                else:
                    continue
        else:
            self.frame_1 = cv2.imread(self.file_path)

    def select_file(self):
        filetypes = (
            ("png", "*.png"),
            ("tiff", "*.tiff"),
            ("mp4", "*.mp4"),
            ("All files", "*.*"),
        )

        self.file_path = tkinter.filedialog.askopenfilename(
            title="Open a file", initialdir="./img", filetypes=filetypes
        )
        if self.file_path.endswith("mp4"):
            self.video_mode = True
        else:
            self.video_mode = False
        self.title(f"Slope Finder ({self.file_path})")

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)

        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    def refresh_image(self, val=0):
        if self.init == 1:
            self.initial_values()
            # self.init = 0
        # self.update_frames()
        # self.video_mode = False
        # print(self.tst_lst)
        # if not self.video_mode:
        self.img = cv2.imread(self.file_path)

        self.update_params()
        # self.save_params()
        _, display_img, _ = self.CVapp.calculate_lines(self.img)
        cv2.imshow("display_img", display_img)
        color_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        widd = self.mycanvas.cget("width")
        resized = self.image_resize(image=color_image, width=int(widd))
        im = Image.fromarray(resized, mode="RGB")
        imgtk = ImageTk.PhotoImage(image=im)
        self.mycanvas.imgref = imgtk
        test_id = self.mycanvas.itemconfig(self.image_id, image=imgtk)

    def process_video(self):
        if True:  # self.video_mode:
            # cap = cv2.VideoCapture(self.file_path)
            cap = cv2.VideoCapture("./vids/avalanche_without_repose.avi")
            while cap.isOpened():
                if True:  # cv2.waitKey(33) & 0xFF == ord("q"):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # try:
                    self.CVapp.current_frame = frame
                    self.refresh_image()
                    # time.sleep(1)
                    # angle_array.append(np.mean(np.abs(angle_array)))
                    # writer.writerow(angle_array)
                    self.frame_2 = frame
                    # except:
                    #     print("Could not obtain line")
                else:
                    continue
        else:
            self.frame_1 = cv2.imread(self.file_path)

    def next_frame(self):
        print("next")

    def play_pause(self):
        print("play")

    def resizer(self, event):
        w, h = event.width - 100, event.height - 100
        self.canvas.config(width=w, height=h)
        # color_image = cv2.cvtColor(self.CVapp.current_frame, cv2.COLOR_BGR2RGB)
        # im = Image.fromarray(color_image, mode="RGB")
        # # image1 = Image.open(self.file_path)
        # resized_image = im.resize((e.width, e.height), Image.Resampling.LANCZOS)
        # new_image = ImageTk.PhotoImage(resized_image)
        # self.canvas.itemconfigure(self.image_id, image=new_image)

    # def resize_image(e):
    #     color_image = cv2.cvtColor(self.CVapp.current_frame, cv2.COLOR_BGR2RGB)
    #     # edge_image = cv2.cvtColor(self.CVapp.current_frame_edge, cv2.COLOR_BGR2RGB)
    #     im = Image.fromarray(color_image, mode="RGB")
    #     # resize the image with width and height of root
    #     resized = self.current_frame.resize((e.width, e.height), Image.ANTIALIAS)

    #     image2 = ImageTk.PhotoImage(resized)
    #     self.canvas.create_image(0, 0, image=image2, anchor="nw")

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
