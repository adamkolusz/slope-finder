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


class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.length = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.imgs = []
        self.frame_number = 0
        _, self.first_frame = self.get_next_frame()
        self.current_frame = self.first_frame

    def get_next_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = rgb_frame
                self.frame_number += 1
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, rgb_frame)
            else:
                return (ret, None)
        else:
            return (ret, None)

    def get_video(self):
        while True:
            ret, frame = self.get_next_frame()
            if not ret:
                break
            self.imgs.append(frame)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


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

        self.frameStyle = GridSettings(0, 0, 0, 0, None, 50, 25, 0, 0, 0, 0, "nswe")
        self.homeStyle = GridSettings(0, 0, 0, 0, None, 50, 25, 5, 5, 10, 10, "nswe")
        self.tabStyle = GridSettings(0, 0, 1, 1, None, 50, 25, 0, 0, 10, 10, "nswe")
        self.labelStyle = GridSettings(0, 0, 0, 0, None, 50, 25, 0, 0, 0, 0, "nsew")
        self.scrollStyle = GridSettings(0, 0, 1, 1, None, 250, 500, 5, 5, 5, 5, "ew")
        self.sliderStyle = GridSettings(0, 0, 0, 0, None, 50, 25, 0, 0, 0, 00, "nsw")
        self.entryStyle = GridSettings(0, 0, 0, 0, None, 50, 25, 0, 0, 0, 0, "nsw")
        self.buttonStyle = GridSettings(0, 0, 2, 0, None, 50, 25, 5, 5, 5, 5, "ns")
        self.actionButtonStyle = GridSettings(
            0, 0, 0, 0, None, 25, 25, 0, 0, 0, 0, "nsw"
        )

        # LAYOUT
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(
            row=self.frameStyle.row,
            column=self.frameStyle.column,
            padx=self.frameStyle.padx,
            pady=self.frameStyle.pady,
            sticky=self.frameStyle.sticky,
        )

        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.button_frame = customtkinter.CTkFrame(
            self.navigation_frame, corner_radius=0, height=120
        )
        self.display_frame = customtkinter.CTkFrame(
            self.navigation_frame, corner_radius=0, height=120
        )
        self.total_array = []
        self.navigation_frame.pack(side="left", fill="both", expand=False)
        self.home_frame.pack(side="left", fill="both", expand=True)
        self.button_frame.pack(side="bottom", fill="y", expand=False)
        self.display_frame.pack(side="bottom", fill="both", expand=False)

        self.tab_name_lst = ["Options", "Filter", "Position"]
        self.video_mode = False
        self.frame_1 = np.array([])
        self.frame_2 = np.array([])
        self.all_angles = np.array([0, 0, 0, 0, 0])
        self.frame_width = 0
        self.frame_height = 0
        self.video_source = "./vids/avalanche_without_repose.avi"
        self.file_name = "example"
        self.vid = None
        self.widgets = {}
        self.pause_vid = False

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
        self.tabview = customtkinter.CTkTabview(self.navigation_frame)
        self.tabview.pack(side="left", fill="both", expand=True)
        self.scrollable_frames = {}

        for i, name in enumerate(self.tab_name_lst):
            self.tabview.add(name)
            self.scrollable_frames[i] = customtkinter.CTkScrollableFrame(
                master=self.tabview.tab(name),
                width=self.scrollStyle.width,
                height=self.scrollStyle.height,
            )
            self.scrollable_frames[i].pack(side="left", fill="both", expand=True)

        self.tabview.set(self.tab_name_lst[0])

        self.labels = {}
        self.sliders = {}
        self.entries = {}
        self.buttons = {}
        self.action_buttons = {}
        self.progress_bars = {}

        self.file_path = "./img/piv/piv2.png"

        self.action_buttons_commands = [
            self.next_frame,
            self.process_video,
            self.jump_to_frame,
            self.refresh_image,
        ]

        self.menu_buttons_commands = [
            self.select_file,
            self.save_file,
            self.load_config,
            self.save_config,
        ]

        self.img_refs = []
        self.init_widgets()

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

        self.image_canvas = ResizingCanvas(
            self.home_frame,
            width=850,
            height=400,
            bg="#000000",
            highlightthickness=2,
        )
        self.image_canvas.pack(fill="both", expand=True)
        self.image_canvas.imgrefs = []

        self.image_id = self.image_canvas.create_image(0, 0, image=self.im, anchor="nw")

        self.CVapp = angle_detector.AngleDetector()
        self.CVapp.load_config()

    def save_file(self):
        pass

    def load_config(self):
        pass

    def save_config(self):
        pass

    def init_widgets(self):
        action_ctr = 0
        menu_btn_ctr = 0
        for i, name in enumerate(self.widgets):
            if self.widgets[name]["type"] == "slider":
                self.labels[name] = customtkinter.CTkLabel(
                    master=self.scrollable_frames[self.widgets[name]["tab"]],
                    text=self.widgets[name]["text"],
                )

                self.labels[name].grid(
                    row=self.labelStyle.row + i * 2,
                    column=self.labelStyle.column,
                    padx=self.labelStyle.padx,
                    pady=self.labelStyle.pady,
                    sticky=self.labelStyle.sticky,
                    columnspan=2,
                )

                self.sliders[name] = customtkinter.CTkSlider(
                    master=self.scrollable_frames[self.widgets[name]["tab"]],
                    from_=self.widgets[name]["min"],
                    to=self.widgets[name]["max"],
                    number_of_steps=self.widgets[name]["step"],
                    command=self.refresh_image,
                    variable=self.tst_lst[name],
                )

                self.sliders[name].grid(
                    row=self.sliderStyle.row + 1 + i * 2, column=self.sliderStyle.column
                )

                self.entries[name] = customtkinter.CTkEntry(
                    master=self.scrollable_frames[self.widgets[name]["tab"]],
                    placeholder_text="CTkEntry",
                    textvariable=self.tst_lst[name],
                    width=self.entryStyle.width,
                    height=self.entryStyle.height,
                )

                # self.entry.bind("<Return>", command=self.get_text)

                self.entries[name].grid(
                    row=self.sliderStyle.row + 1 + i * 2,
                    column=self.sliderStyle.column + 1,
                    padx=self.entryStyle.padx,
                    pady=self.entryStyle.pady,
                )
            elif (
                self.widgets[name]["type"] == "button"
                and self.widgets[name]["tab"] != 10
            ):
                self.buttons[name] = customtkinter.CTkButton(
                    master=self.scrollable_frames[self.widgets[name]["tab"]],
                    width=self.buttonStyle.width,
                    height=self.buttonStyle.height,
                    text=self.widgets[name]["text"],
                    command=self.menu_buttons_commands[menu_btn_ctr],
                )
                # Union[Callable[[], None], None]
                self.buttons[name].grid(
                    row=self.buttonStyle.row + i * 2,
                    column=self.buttonStyle.column,
                    padx=self.buttonStyle.padx,
                    pady=self.buttonStyle.pady,
                    sticky=self.buttonStyle.sticky,
                    columnspan=self.buttonStyle.columnspan,
                )
                menu_btn_ctr += 1
            elif self.widgets[name]["type"] == "action_button":
                self.action_buttons[name] = customtkinter.CTkButton(
                    master=self.button_frame,
                    text=str(self.widgets[name]["text"]),
                    width=self.actionButtonStyle.width,
                    height=self.actionButtonStyle.height,
                    command=self.action_buttons_commands[action_ctr],
                )
                action_ctr += 1
                self.action_buttons[name].pack(side="left", fill="both", expand=True)
            elif self.widgets[name]["type"] == "display_label":
                if name == "percent_label":
                    self.progress_bars[name] = customtkinter.CTkProgressBar(
                        master=self.display_frame
                    )
                    self.progress_bars[name].pack(
                        side="bottom", fill="both", expand=True
                    )
                    self.progress_bars[name].set(0)

                self.labels[name] = customtkinter.CTkLabel(
                    master=self.display_frame,
                    text=self.widgets[name]["text"],
                )

                self.labels[name].pack(side="bottom", fill="both", expand=True)

                # if name == "frame_number":
                #     self.progress_bars[name] = customtkinter.CTkProgressBar(
                #         master=self.display_frame
                #     )
                #     self.progress_bars[name].pack(
                #         side="bottom", fill="both", expand=True
                #     )
                #     self.progress_bars[name].set(0)

                # self.labels[name] = customtkinter.CTkLabel(
                #     master=self.display_frame,
                #     text=self.widgets[name]["text"],
                # )

                self.labels[name].pack(side="bottom", fill="both", expand=True)

    def update_sliders(self):
        self.update_widget_parameters()
        for name in self.sliders:
            self.sliders[name].configure(
                from_=self.widgets[name]["min"],
                to=self.widgets[name]["max"],
                number_of_steps=self.widgets[name]["step"],
            )

    def update_widget_parameters(self):
        self.update_frame_size()
        self.widgets["top_boundary"]["max"] = self.frame_height
        self.widgets["top_boundary"]["step"] = self.frame_height

        self.widgets["bottom_boundary"]["max"] = self.frame_height
        self.widgets["bottom_boundary"]["step"] = self.frame_height

        self.widgets["left_boundary"]["max"] = self.frame_width
        self.widgets["left_boundary"]["step"] = self.frame_width

        self.widgets["right_boundary"]["max"] = self.frame_width
        self.widgets["right_boundary"]["step"] = self.frame_width

        self.widgets["middle_line_position"]["max"] = self.frame_width
        self.widgets["middle_line_position"]["step"] = self.frame_width

    def initial_values(self):
        self.init = 0
        for i, name in enumerate(self.widgets):
            if self.widgets[name]["type"] == "slider":
                self.tst_lst[name].set(self.widgets[name]["value"])

    def save_params(self):
        pass
        # with open("config/data.json", "w") as fp:
        #     json.dump(self.widgets, fp, indent=4)
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
            ("All files", "*.*"),
            ("png", "*.png"),
            ("tiff", "*.tiff"),
            ("mp4", "*.mp4"),
            ("avi", "*.avi"),
        )

        self.file_path = tkinter.filedialog.askopenfilename(
            title="Open a file", initialdir="./vids", filetypes=filetypes
        )
        if self.file_path.endswith("mp4") or self.file_path.endswith("avi"):
            self.video_mode = True
            self.video_source = self.file_path
            self.file_name = self.file_path.split("/")[-1].split(".")[0]
            self.vid = MyVideoCapture(self.video_source)
            self.update_params()
            print("Video mode!")
        else:
            self.video_mode = False
            self.video_source = None
            self.frame_1 = cv2.imread(self.file_path)
            self.update_params()
        self.total_array = []
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

    def calculate_angle_values(self, input_array1, input_array2, total_array):
        left, right = input_array1, input_array2
        avg = (np.abs(left) + np.abs(right)) / 2
        total_avg = np.average(total_array[2])
        std_dev = np.std(total_array[3])
        # total_array.append([left, right, avg, total_avg, std_dev])
        # total_array = np.append(total_array, [left, right, avg, total_avg, std_dev])
        return left, right, avg, total_avg, std_dev

    def refresh_image(self, val=0):
        if self.init == 1:
            self.initial_values()
        if not self.video_mode:
            self.img = cv2.imread(self.file_path)
            self.CVapp.current_frame = self.img
        else:
            self.img = self.vid.current_frame
            percentage = (self.vid.frame_number - 1) / (self.vid.length - 1)
            self.labels["percent_label"].configure(
                text=f"Frame: {int(self.vid.frame_number)}/{int(self.vid.length)}({percentage*100:.2f}%)"
            )
            self.progress_bars["percent_label"].set(percentage)
        self.update_params()
        self.save_params()
        angle_array, _, _ = self.CVapp.calculate_lines(self.img)
        left, right, avg, total_avg, std_dev = self.calculate_angle_values(
            angle_array[0], angle_array[1], self.all_angles
        )
        self.all_angles = np.append(
            self.all_angles, [left, right, avg, total_avg, std_dev]
        )
        # print(f"{self.all_angles[:-1]=}")
        self.labels["angle_label"].configure(
            text=f"Angles: {left:.1f}°-{right:.1f}° (avg: {avg:.1f}°, σ: {std_dev:.1f}°)"
        )
        color_image = cv2.cvtColor(self.CVapp.current_frame, cv2.COLOR_BGR2RGB)
        canvas_width = self.image_canvas.cget("width")
        canvas_height = self.image_canvas.cget("height")
        frame_size = np.shape(self.img)
        if frame_size[1] > frame_size[0]:
            resized = self.image_resize(image=color_image, width=int(canvas_width))
        else:
            resized = self.image_resize(image=color_image, height=int(canvas_height))
        im = Image.fromarray(resized, mode="RGB")
        imgtk = ImageTk.PhotoImage(image=im)

        self.image_canvas.imgref = imgtk
        test_id = self.image_canvas.itemconfig(
            self.image_id, image=self.image_canvas.imgref
        )
        return angle_array

    def update_frame_size(self):
        frame_dim = np.shape(self.img)
        # print(f"{frame_dim=}")
        self.frame_width = frame_dim[1]
        self.frame_height = frame_dim[0]

    # def get_frame(self):
    #     ret, frame = self.vid.read()
    #     if ret:
    #         return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     else:
    #         return (ret, None)

    def start_processing(self):
        pass

    def process_video(self):
        # self.video_mode:
        # self.video_mode = True

        self.pause_vid = not self.pause_vid
        while not self.pause_vid:
            ret, frame = self.vid.get_next_frame()
            if not ret:
                break
            self.frame_1 = frame
            current_angles = self.refresh_image()
            self.total_array.append(current_angles)
            self.update()
            # angle_array.append(np.mean(np.abs(angle_array)))
            # writer.writerow(angle_array)
            # self.frame_2 = self.frame_1
            # self.after(15, self.process_video)

            # except:
            #     print("Could not obtain line")
        # self.video_mode = False

        self.CVapp.save_to_csv(
            self.total_array, True, dir="data/exp/", name=str(self.file_name)
        )

    def jump_to_frame(self):
        self.vid.frame_number = self.vid.frame_number + 250
        self.vid.vid.set(cv2.CAP_PROP_POS_FRAMES, self.vid.frame_number)

    def next_frame(self):
        self.update_sliders()
        self.CVapp.save_config(self.widgets)

    def play_pause(self):
        print("play")
        self.pause_vid = not self.pause_vid

    # def resizer(self, event):
    #     w, h = event.width - 100, event.height - 100
    #     self.canvas.config(width=w, height=h)
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
    import cProfile

    app = App()
    cProfile.run("app.mainloop()", "profiling_output.dat")
