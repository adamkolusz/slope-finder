import numpy as np
import cv2

import math
import sys
import json
import csv
import time
import glob
import os

from pathlib import Path
from collections import defaultdict


class AngleDetector:
    def __init__(self):
        self.loaded = 0
        self.data_json = self.load_config()

        self.kernel_size = self.data_json["kernel_size"]["value"]
        self.low_threshold = self.data_json["low_threshold"]["value"]
        self.high_threshold = self.data_json["high_threshold"]["value"]
        self.rho = self.data_json["rho"]["value"]
        self.theta = self.data_json["theta"]["value"]
        # angular resolution in radians of the Hough grid
        self.threshold = self.data_json["threshold"]["value"]
        # minimum number of votes (intersections in Hough grid cell)
        self.min_line_length = self.data_json["min_line_length"]["value"]
        # minimum number of pixels making up a line
        self.max_line_gap = self.data_json["max_line_gap"]["value"]
        # maximum gap in pixels between connectable line segments
        self.current_image_mode = self.data_json["current_image_mode"]["value"]
        self.scale_percent = self.data_json["scale_percent"]["value"]
        self.line_visibility = self.data_json["line_visibility"]["value"]
        self.image_window_name = self.data_json["image_window_name"]["value"]
        self.trackbar_window_name = self.data_json["trackbar_window_name"]["value"]
        self.top_boundary = self.data_json["top_boundary"]["value"]
        self.bottom_boundary = self.data_json["bottom_boundary"]["value"]
        self.left_boundary = self.data_json["left_boundary"]["value"]
        self.right_boundary = self.data_json["right_boundary"]["value"]
        self.middle_line_position = self.data_json["middle_line_position"]["value"]
        self.hsv_hue_bottom = self.data_json["hsv_hue_bottom"]["value"]
        self.hsv_hue_top = self.data_json["hsv_hue_top"]["value"]
        self.hsv_saturation_bottom = self.data_json["hsv_saturation_bottom"]["value"]
        self.hsv_saturation_top = self.data_json["hsv_saturation_top"]["value"]
        self.hsv_value_bottom = self.data_json["hsv_value_bottom"]["value"]
        self.hsv_value_top = self.data_json["hsv_value_top"]["value"]
        self.current_frame = np.array([])
        self.save_state = 0
        self.enable_cropping = 0
        self.max_thresh = 379
        self.draw_windows = 0
        self.thresh = 50
        self.path = "C:/Users/adamk/Projects/cone_detection/img/fluent/Results"
        # self.current_frame = self.load_image(self.path)

    def load_config(self):
        with open("config/data.json", "r") as f:
            data_json = json.load(f)
        return data_json

    def refresh_view(self):
        try:
            _, _, _ = self.calculate_lines(self.current_frame)
        except cv2.error:
            print("#", end="")

    def change_scale_image(self, val):
        # global scale_percent
        self.scale_percent = val
        self.refresh_view()

    def change_rho(self, val):
        # global rho
        self.rho = val
        self.refresh_view()

    def change_theta(self, val):
        # global theta
        self.theta = np.pi / val
        self.refresh_view()

    def change_threshold(self, val):
        # global threshold
        self.threshold = val
        self.refresh_view()

    def change_line_visibility(self, val):
        # global line_visibility
        self.line_visibility = val
        self.refresh_view()

    def change_min_line_length(self, val):
        # global min_line_length
        self.min_line_length = val
        self.refresh_view()

    def change_low_threshold(self, val):
        # global low_threshold
        self.low_threshold = val
        self.refresh_view()

    def change_high_threshold(self, val):
        # global high_threshold
        self.high_threshold = val
        self.refresh_view()

    def change_max_line_gap(self, val):
        # global max_line_gap
        self.max_line_gap = val
        self.refresh_view()

    def change_top_boundary(self, val):
        # global top_boundary
        self.top_boundary = val
        self.refresh_view()

    def change_bottom_boundary(self, val):
        # global bottom_boundary
        self.bottom_boundary = val
        self.refresh_view()

    def change_left_boundary(self, val):
        # global left_boundary
        self.left_boundary = val
        self.refresh_view()

    def change_right_boundary(self, val):
        # global right_boundary
        self.right_boundary = val
        self.refresh_view()

    def change_current_image_mode(self, val):
        # global current_image_mode
        self.current_image_mode = val
        self.refresh_view()

    def change_hsv_hue_bottom(self, val):
        # global hsv_hue_bottom
        self.hsv_hue_bottom = val
        self.refresh_view()

    def change_hsv_hue_top(self, val):
        # global hsv_hue_top
        self.hsv_hue_top = val
        self.refresh_view()

    def change_hsv_saturation_bottom(self, val):
        # global hsv_saturation_bottom
        self.hsv_saturation_bottom = val
        self.refresh_view()

    def change_hsv_saturation_top(self, val):
        # global hsv_saturation_top
        self.hsv_saturation_top = val
        self.refresh_view()

    def change_hsv_value_bottom(self, val):
        # global hsv_value_bottom
        self.hsv_value_bottom = val
        self.refresh_view()

    def change_hsv_value_top(self, val):
        # global hsv_value_top
        self.hsv_value_top = val
        self.refresh_view()

    def change_middle_line_position(self, val):
        # global middle_line_position
        self.middle_line_position = val
        self.refresh_view()

    def change_cropping_state(self, val):
        # global enable_cropping
        self.enable_cropping = val
        self.refresh_view()

    def change_save_state(self, val):
        # global save_state
        self.save_state = val

    def get_angle_of_line(self, x1, y1, x2, y2):
        x_diff = x2 - x1
        y_diff = y2 - y1
        return np.rad2deg(math.atan2(x_diff, y_diff))

    def split_angles_to_signed(self, input_array):
        d = defaultdict(list)
        for num in input_array:
            if num < 0:
                d["neg"].append(num)
            elif num > 0:
                d["pos"].append(num)
        return d

    def filter_hsv(self, src_image):
        hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
        # set the lower and upper bounds for the green hue
        lower_green = np.array(
            [self.hsv_hue_bottom, self.hsv_saturation_bottom, self.hsv_value_bottom]
        )
        upper_green = np.array(
            [self.hsv_hue_top, self.hsv_saturation_top, self.hsv_value_top]
        )

        # create a mask for green colour using inRange function
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # perform bitwise and on the original image arrays using the mask
        res = cv2.bitwise_and(hsv, hsv, mask=mask)
        resultant = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        return resultant

    def limit_angles_to_boundary(self, src_array):
        output_array = []
        for line in src_array:
            for x1, y1, x2, y2 in line:
                if (
                    self.top_boundary < y1 < self.bottom_boundary
                    and self.top_boundary < y2 < self.bottom_boundary
                ):
                    angle = self.get_angle_of_line(y1, x1, y2, x2)
                    if (-75 < angle < -15) or (15 < angle < 75):
                        output_array.append([angle, x1, y1, x2, y2])
        return output_array

    def draw_boundaries(self, src_image2):
        src_image = np.copy(src_image2)
        width = int(src_image.shape[1])
        height = int(src_image.shape[0])
        cv2.line(
            src_image, (0, self.top_boundary), (width, self.top_boundary), (0, 0, 0), 3
        )
        cv2.line(
            src_image,
            (0, self.top_boundary),
            (width, self.top_boundary),
            (255, 255, 255),
            1,
        )

        cv2.line(
            src_image,
            (0, self.bottom_boundary),
            (width, self.bottom_boundary),
            (0, 0, 0),
            3,
        )
        cv2.line(
            src_image,
            (0, self.bottom_boundary),
            (width, self.bottom_boundary),
            (255, 255, 255),
            1,
        )

        cv2.line(
            src_image,
            (self.left_boundary, self.top_boundary),
            (self.left_boundary, self.bottom_boundary),
            (100, 0, 255),
            3,
        )
        # cv2.line(src_image, (left_boundary, bottom_boundary), (left_boundary, height), (0, 255, 0), 3)

        cv2.line(
            src_image,
            (self.right_boundary, self.top_boundary),
            (self.right_boundary, self.bottom_boundary),
            (255, 0, 100),
            3,
        )

        cv2.line(
            src_image,
            (self.middle_line_position, self.top_boundary),
            (self.middle_line_position, self.bottom_boundary),
            (255, 50, 255),
            3,
        )
        return src_image

    def draw_angled_lines(self, src_edge_image, dst_colour_image, src_array):
        for angle, x1, y1, x2, y2 in src_array:
            color = np.random.randint(0, 255, size=(3,))
            color = (int(color[0]), int(color[1]), int(color[2]))
            cv2.line(dst_colour_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.line(dst_colour_image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                dst_colour_image,
                f"{angle:.2f}",
                (round((x1 + x2) / 2), round((y1 + y2) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                lineType=cv2.LINE_AA,
                thickness=3,
            )
            cv2.putText(
                dst_colour_image,
                f"{angle:.2f}",
                (round((x1 + x2) / 2), round((y1 + y2) / 2)),
                cv2.FONT_ITALIC,
                0.75,
                (0, 0, 0),
                lineType=cv2.LINE_AA,
                thickness=3,
            )
            cv2.putText(
                dst_colour_image,
                f"{angle:.2f}",
                (round((x1 + x2) / 2), round((y1 + y2) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                lineType=cv2.LINE_AA,
                thickness=2,
            )

    def find_aor(self, src_colour_image2, src_edge_image2):
        # src_colour_image = np.copy(src_colour_image2)
        # src_edge_image = np.copy(src_edge_image2)
        src_colour_image = src_colour_image2
        src_edge_image = src_edge_image2
        xy_coords = np.flip(np.column_stack(np.where(src_edge_image > 0)), axis=1)
        # xy_array = np.flip(xy_coords[:,0],xy_coords[:,1])
        # print(xy_coords)
        # print(np.shape(xy_coords))
        vx, vy, x, y = cv2.fitLine(xy_coords, cv2.DIST_WELSCH, 0, 0.1, 0.1)
        lefty = int(np.round((-x * vy / vx) + y))
        righty = int(np.round(((np.size(src_edge_image, 1) - x) * vy / vx) + y))
        point1 = (np.size(src_edge_image, 1) - 1, righty)
        point2 = (0, lefty)
        # print(f"{point1=}")
        # print(f"{point2=}")
        cv2.line(src_edge_image, point1, point2, (100, 255, 100), 3)
        cv2.line(src_colour_image, point1, point2, (100, 255, 100), 3)
        angle = self.get_angle_of_line(point2[1], point2[0], point1[1], point1[0])
        # print(angle)
        return angle, [point1, point2]

    def split_img(self, src_image):
        # middle_line = int(round(src_image.shape[1] / 2))
        # global middle_line_position
        left_crop = src_image[
            self.top_boundary : self.bottom_boundary,
            self.left_boundary : self.middle_line_position,
        ]
        right_crop = src_image[
            self.top_boundary : self.bottom_boundary,
            self.middle_line_position : self.right_boundary,
        ]
        return left_crop, right_crop

    def crop_image(self, src_image, top, bot, left, right):
        crop_img = src_image[top:bot, left:right]
        return crop_img

    def write_angles_on_img(self, src_array2, src_img2):
        src_array = np.copy(src_array2)
        src_img = np.copy(src_img2)
        text_right = f"Right slope mean: {src_array[0]:.2f}"
        text_left = f"Left slope mean: {src_array[1]:.2f}"

        mean_value = (abs(src_array[0]) + abs(src_array[1])) / 2
        text_middle = f"{mean_value:.2f}"
        text_final = f"{src_array[1]:.2f} : {src_array[0]:.2f} : {mean_value:.2f}"
        move_up = -600
        move_horiz = 100
        cv2.putText(
            src_img,
            text_left,
            (0 + move_horiz, src_img.shape[0] - 50 + move_up),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            lineType=cv2.LINE_AA,
            thickness=3,
        )
        cv2.putText(
            src_img,
            text_right,
            (0 + move_horiz, src_img.shape[0] - 25 + move_up),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            lineType=cv2.LINE_AA,
            thickness=3,
        )
        cv2.putText(
            src_img,
            text_middle,
            (340 + move_horiz, src_img.shape[0] - 25 + move_up),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            lineType=cv2.LINE_AA,
            thickness=4,
        )

        cv2.putText(
            src_img,
            text_left,
            (0 + move_horiz, src_img.shape[0] - 50 + move_up),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            lineType=cv2.LINE_AA,
            thickness=1,
        )
        cv2.putText(
            src_img,
            text_right,
            (0 + move_horiz, src_img.shape[0] - 25 + move_up),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            lineType=cv2.LINE_AA,
            thickness=1,
        )
        cv2.putText(
            src_img,
            text_middle,
            (340 + move_horiz, src_img.shape[0] - 25 + move_up),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            lineType=cv2.LINE_AA,
            thickness=2,
        )
        return src_img

    def find_lines_and_draw_on_images(self, src_edge_image2, dst_colour_image2):
        src_edge_image = src_edge_image2
        dst_colour_image = dst_colour_image2
        lines = cv2.HoughLinesP(
            src_edge_image,
            self.rho,
            self.theta,
            self.threshold,
            np.array([]),
            self.min_line_length,
            self.max_line_gap,
        )

        line_number = 0
        print(f"#" * 30)

        # all_found_angles_list = np.zeros((1,))
        # print(f"eee {positive_angle_lines}")
        all_found_angles_list = []
        angles_with_coords = []
        # np.zeros([])
        # print(f"line1 = {all_found_angles_list}")
        # print(f"{fit_line=}")
        # print(f"{lines=})
        angles_with_coords = self.limit_angles_to_boundary(lines)
        print(f"{angles_with_coords=}")
        self.draw_angled_lines(src_edge_image, dst_colour_image, angles_with_coords)
        self.draw_boundaries(dst_colour_image)
        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #
        #
        #         current_angle
        #
        #         # all_found_angles_list = np.vstack([all_found_angles_list, current_angle])
        #         if (-90 < current_angle < -5) or (5 < current_angle < 90):
        #             all_found_angles_list.append(current_angle)
        #             angles_with_coords.append([current_angle, x1, y1, x2, y2])
        #             # print(f"Angle of line [{line_number}]: {current_angle:.2f}")
        #             line_number += 1

        slice = [item[0] for item in angles_with_coords]
        print(f"Found angles: {slice}")
        angle_dict = self.split_angles_to_signed(slice)

        positive_angles = angle_dict["pos"]
        negative_angles = angle_dict["neg"]
        mean_pos_angles = np.mean(positive_angles)
        mean_neg_angles = np.mean(negative_angles)
        angle_of_repose = np.mean([mean_pos_angles, -mean_neg_angles])

        text_right = f"Right slope mean: {mean_pos_angles:.2f}"
        text_left = f"Left slope mean: {mean_neg_angles:.2f}"
        text_middle = f"{angle_of_repose:.2f}"
        cv2.putText(
            dst_colour_image,
            text_left,
            (0, dst_colour_image.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            lineType=cv2.LINE_AA,
            thickness=3,
        )
        cv2.putText(
            dst_colour_image,
            text_right,
            (0, dst_colour_image.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            lineType=cv2.LINE_AA,
            thickness=3,
        )
        cv2.putText(
            dst_colour_image,
            text_middle,
            (340, dst_colour_image.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            lineType=cv2.LINE_AA,
            thickness=4,
        )

        cv2.putText(
            dst_colour_image,
            text_left,
            (0, dst_colour_image.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            lineType=cv2.LINE_AA,
            thickness=1,
        )
        cv2.putText(
            dst_colour_image,
            text_right,
            (0, dst_colour_image.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            lineType=cv2.LINE_AA,
            thickness=1,
        )
        cv2.putText(
            dst_colour_image,
            text_middle,
            (340, dst_colour_image.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            lineType=cv2.LINE_AA,
            thickness=2,
        )
        print(text_right)
        print(text_left)
        print(text_middle)
        # split_angles_to_signed(all_found_angles_list)
        # Draw the lines on the  image
        # lines_edges = cv2.addWeighted(dst_colour_image, 1, line_image, 1, 0)
        # horizontalAppendedImg = np.hstack((gray, edges))
        img_to_draw = np.zeros((100, 100, 3), dtype=np.uint8)
        # print(lines)

    def load_image(self, path):
        img_colour = cv2.imread(path)
        return img_colour

    def filter_edges_on_image(self, img_colour2):
        img_colour = np.copy(img_colour2)
        gray = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)
        # line_image = np.copy(img_colour) * 0  # creating a blank to draw lines on
        blur_gray = cv2.GaussianBlur(
            gray, (int(self.kernel_size), int(self.kernel_size)), 0
        )
        edges = cv2.Canny(
            blur_gray, self.low_threshold, self.high_threshold, self.kernel_size
        )
        return edges

    def resize_image(self, img):
        width = int(img.shape[1] * self.scale_percent / 100)
        height = int(img.shape[0] * self.scale_percent / 100)
        dim = (width, height)
        img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return img_resized

    def select_from_images_and_draw(self, img1, img2, number):
        if number == 0:
            cv2.imshow(self.image_window_name, img1)
        else:
            cv2.imshow(self.image_window_name, img2)

    def save_img_to_path(self, src_img, name):
        global save_state
        if save_state == 1:
            cv2.imwrite(name, src_img)

    # def refresh_view():
    #     global current_frame
    #     calculate_lines(current_frame)

    def merge_images(self, img_left, img_right):
        horizontal_img = np.hstack((img_left, img_right))
        return horizontal_img

    def draw_angled_line(self, img, points, colour, thickness):
        cv2.line(img, points[0], points[1], colour, thickness)

    def calculate_lines(self, input_image):  # load_image("img/img_14.png")):
        # colour_image = load_image("temp.png")
        # global current_frame
        # cv2.imwrite("temp.png", current_frame)

        self.kernel_size = self.data_json["kernel_size"]["value"]
        self.low_threshold = self.data_json["low_threshold"]["value"]
        self.high_threshold = self.data_json["high_threshold"]["value"]
        self.rho = self.data_json["rho"]["value"]
        self.theta = self.data_json["theta"]["value"]
        # angular resolution in radians of the Hough grid
        self.threshold = self.data_json["threshold"]["value"]
        # minimum number of votes (intersections in Hough grid cell)
        self.min_line_length = self.data_json["min_line_length"]["value"]
        # minimum number of pixels making up a line
        self.max_line_gap = self.data_json["max_line_gap"]["value"]
        # maximum gap in pixels between connectable line segments
        self.current_image_mode = self.data_json["current_image_mode"]["value"]
        self.scale_percent = self.data_json["scale_percent"]["value"]
        self.line_visibility = self.data_json["line_visibility"]["value"]
        self.image_window_name = self.data_json["image_window_name"]["value"]
        self.trackbar_window_name = self.data_json["trackbar_window_name"]["value"]
        self.top_boundary = self.data_json["top_boundary"]["value"]
        self.bottom_boundary = self.data_json["bottom_boundary"]["value"]
        self.left_boundary = self.data_json["left_boundary"]["value"]
        self.right_boundary = self.data_json["right_boundary"]["value"]
        self.middle_line_position = self.data_json["middle_line_position"]["value"]
        self.hsv_hue_bottom = self.data_json["hsv_hue_bottom"]["value"]
        self.hsv_hue_top = self.data_json["hsv_hue_top"]["value"]
        self.hsv_saturation_bottom = self.data_json["hsv_saturation_bottom"]["value"]
        self.hsv_saturation_top = self.data_json["hsv_saturation_top"]["value"]
        self.hsv_value_bottom = self.data_json["hsv_value_bottom"]["value"]
        self.hsv_value_top = self.data_json["hsv_value_top"]["value"]

        prev_frame = np.copy(input_image)

        # global current_frame
        self.current_frame = prev_frame
        # cv2.imshow("1", input_image)
        colour_image = np.copy(input_image)
        # colour_image = current_frame[:]
        # save_img_to_path(input_image, "temp.png")
        # cv2.imwrite("temp.png", input_image)
        # global current_frame
        # global current_frameq
        # colour_image = filter_hsv(current_frame)
        # colour_image = current_frame
        # colour_image = input_image[:]
        edge_image = self.filter_edges_on_image(colour_image)
        edge_left_crop, edge_right_crop = self.split_img(edge_image)
        colour_left_crop, colour_right_crop = self.split_img(colour_image)
        left_angle, left_points_array = self.find_aor(colour_left_crop, edge_left_crop)
        right_angle, right_points_array = self.find_aor(
            colour_right_crop, edge_right_crop
        )
        # draw_angled_line(colour_image, left_points_array, (100,255,100), 3)

        # colourmerge_images(colour_left_crop2, colour_right_crop2)
        print(f"Left: {left_angle:.2f} Right: {right_angle:.2f}")
        if self.line_visibility == 1:
            self.find_lines_and_draw_on_images(edge_image, colour_image)
        edge_image2 = self.draw_boundaries(edge_image)
        colour_image2 = self.draw_boundaries(colour_image)
        edge_image3 = self.write_angles_on_img([left_angle, right_angle], edge_image2)
        colour_image3 = self.write_angles_on_img(
            [left_angle, right_angle], colour_image2
        )

        if self.enable_cropping == 1:
            cropped_edge_image = self.crop_image(
                edge_image3,
                self.top_boundary,
                self.bottom_boundary,
                self.left_boundary,
                self.right_boundary,
            )
            cropped_colour_image = self.crop_image(
                colour_image3,
                self.top_boundary,
                self.bottom_boundary,
                self.left_boundary,
                self.right_boundary,
            )
            resized_edge_image = self.resize_image(cropped_edge_image)
            resized_colour_image = self.resize_image(cropped_colour_image)
        else:
            resized_edge_image = self.resize_image(edge_image3)
            resized_colour_image = self.resize_image(colour_image3)

        # cv2.imshow("3", resized_colour_image)

        # select_from_images_and_draw(ww, ww, current_image_mode)
        # cv2.imshow(image_window_name, colour_image)qqqqqqqqqq

        if self.draw_windows == 1:
            self.select_from_images_and_draw(
                resized_edge_image, resized_colour_image, self.current_image_mode
            )
        # global current_frame
        # current_frame = input_image
        # cv2.imshow("colour_image3", colour_image3)
        return [left_angle, right_angle], colour_image3, prev_frame

    def save_to_csv(
        self, input_array, save, dir="test/", name=time.strftime("dem-%Y%m%d-%H%M%S")
    ):
        if save == True:
            csv_name = dir + name + ".csv"
            with open(csv_name, "w", encoding="UTF8", newline="") as f1:
                writer = csv.writer(
                    f1,
                    delimiter=",",
                    lineterminator="\n",
                )
                for i in range(len(input_array)):
                    writer.writerow(input_array[i])
        else:
            print("Data not saved to csv!")

    def generate_video_from_path(self, path, video_name, fps):
        file_list = list_files_in_dir(path)
        frame = cv2.imread(os.path.join(path, file_list[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, fps, (width, height))

        for image in file_list:
            video.write(cv2.imread(os.path.join(path, image)))
        cv2.destroyAllWindows()
        video.release()

    def list_files_in_dir(self, path):
        file_list = []
        for fname in os.listdir(path):
            newpath = os.path.join(path, fname)
            if os.path.isdir(newpath):
                # skip directories
                continue
            file_list.append(fname)
        file_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        return file_list

    def repose_calculation_from_images(self, path):
        # for file in glob.glob(f"{path}/*.png"):
        total_array = []
        mean_angle = []
        file_list = self.list_files_in_dir(path)
        for count, file in enumerate(file_list):
            image = cv2.imread(f"{path}/{file}")
            # if cv2.waitKey(0):
            if cv2.waitKey(0) or 0xFF == ord("q"):
                try:
                    self.current_frame = image
                    angle_array, img, _ = self.calculate_lines(image)

                    new_name = "repose_" + os.path.splitext(file)[0] + ".png"
                    if self.save_state == 1:
                        self.save_img_to_path(img, path + "/repose/" + new_name)
                        Path(f"{path}/repose/").mkdir(parents=True, exist_ok=True)
                        print(f"saved: {path + '/repose/'+ new_name}")
                    print(new_name)
                    mean_angle = [np.mean(np.abs(angle_array))]
                    print(mean_angle)
                    mean_angle.append(new_name)
                    total_array.append(mean_angle)
                    # writer.writerow(angle_array)
                except:
                    print("Could not obtain line")
            else:
                continue
        print(total_array)
        print(f"len={len(total_array)}")
        self.save_to_csv(total_array, True)
        pass

    def repose_calculation_from_video(self, path):
        global current_frame
        dir = "data/"
        mode = 0
        try:
            cap = cv2.VideoCapture(path)
        except:
            mode = 1
        csv_name = dir + time.strftime("dem-%Y%m%d-%H%M%S") + ".csv"
        with open(csv_name, "w", encoding="UTF8", newline="") as f1:
            writer = csv.writer(
                f1,
                delimiter=",",
                lineterminator="\n",
            )
            if True:  # mode == 0:
                while cap.isOpened():
                    if True:  # cv2.waitKey(0) & 0xFF == ord('q'):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        try:
                            current_frame = frame
                            angle_array, _ = self.calculate_lines(current_frame)
                            angle_array.append(np.mean(np.abs(angle_array)))
                            writer.writerow(angle_array)
                        except:
                            print("Could not obtain line")
                    else:
                        continue
            # else:
            #     for file in glob.glob("Figures/*.png"):
            #         print(file)
            #         image = cv2.imread(file)
            #         if cv2.waitKey(0) & 0xFF == ord('q'):
            #             try:
            #                 current_frame = image
            #                 angle_array = calculate_lines(current_frame)
            #                 angle_array.append(file)
            #                 writer.writerow(angle_array)
            #             except:
            #                 print("Could not obtain line")
            #         else:
            #             continue

    def main(self):
        cv2.namedWindow(self.trackbar_window_name, cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(self.trackbar_window_name, 600, 1280)
        cv2.namedWindow(self.image_window_name, cv2.WINDOW_AUTOSIZE)

        # new_path = input("Path:/n")
        new_path = "img/Ex1.png"

        # global current_frame
        current_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        current_frame = self.load_image(new_path)

        cv2.createTrackbar(
            "Mid:",
            self.trackbar_window_name,
            self.middle_line_position,
            4096,
            self.change_middle_line_position,
        )
        cv2.createTrackbar(
            "%:",
            self.trackbar_window_name,
            self.scale_percent,
            400,
            self.change_scale_image,
        )
        cv2.createTrackbar(
            "OFF/ON:",
            self.trackbar_window_name,
            self.line_visibility,
            1,
            self.change_line_visibility,
        )
        cv2.createTrackbar(
            "MODE:",
            self.trackbar_window_name,
            self.current_image_mode,
            1,
            self.change_current_image_mode,
        )
        cv2.createTrackbar(
            "low_tres:",
            self.trackbar_window_name,
            self.low_threshold,
            self.max_thresh,
            self.change_low_threshold,
        )
        cv2.createTrackbar(
            "high_tres:",
            self.trackbar_window_name,
            self.high_threshold,
            self.max_thresh,
            self.change_high_threshold,
        )
        cv2.createTrackbar(
            "min_len:",
            self.trackbar_window_name,
            self.min_line_length,
            self.max_thresh,
            self.change_min_line_length,
        )
        cv2.createTrackbar(
            "gap:",
            self.trackbar_window_name,
            self.max_line_gap,
            250,
            self.change_max_line_gap,
        )
        cv2.createTrackbar(
            "rho:", self.trackbar_window_name, self.rho, 60, self.change_rho
        )
        cv2.createTrackbar(
            "thres:",
            self.trackbar_window_name,
            self.threshold,
            60,
            self.change_threshold,
        )
        cv2.createTrackbar(
            "theta:", self.trackbar_window_name, int(self.theta), 600, self.change_theta
        )

        cv2.createTrackbar(
            "top_boundary:",
            self.trackbar_window_name,
            self.top_boundary,
            2000,
            self.change_top_boundary,
        )
        cv2.createTrackbar(
            "bottom_boundary:",
            self.trackbar_window_name,
            self.bottom_boundary,
            2000,
            self.change_bottom_boundary,
        )
        cv2.createTrackbar(
            "left_boundary:",
            self.trackbar_window_name,
            self.left_boundary,
            2000,
            self.change_left_boundary,
        )
        cv2.createTrackbar(
            "right_boundary:",
            self.trackbar_window_name,
            self.right_boundary,
            4000,
            self.change_right_boundary,
        )

        cv2.createTrackbar(
            "HueT:",
            self.trackbar_window_name,
            self.hsv_hue_top,
            180,
            self.change_hsv_hue_top,
        )
        cv2.createTrackbar(
            "HueB:",
            self.trackbar_window_name,
            self.hsv_hue_bottom,
            180,
            self.change_hsv_hue_bottom,
        )
        cv2.createTrackbar(
            "SatT:",
            self.trackbar_window_name,
            self.hsv_saturation_top,
            255,
            self.change_hsv_saturation_top,
        )
        cv2.createTrackbar(
            "SatB:",
            self.trackbar_window_name,
            self.hsv_saturation_bottom,
            255,
            self.change_hsv_saturation_bottom,
        )
        cv2.createTrackbar(
            "ValT:",
            self.trackbar_window_name,
            self.hsv_value_top,
            255,
            self.change_hsv_value_top,
        )
        cv2.createTrackbar(
            "ValB:",
            self.trackbar_window_name,
            self.hsv_value_bottom,
            255,
            self.change_hsv_value_bottom,
        )

        cv2.createTrackbar(
            "Save",
            self.trackbar_window_name,
            self.save_state,
            1,
            self.change_save_state,
        )
        cv2.createTrackbar(
            "Crop",
            self.trackbar_window_name,
            self.enable_cropping,
            1,
            self.change_cropping_state,
        )

        print("\nLoaded!")

        fps = 30
        # current_frame = load_image(path)
        # while cv2.waitKey(0) & 0xFF == ord('q'):
        #     angle_array, img, _ = calculate_lines(current_frame)
        self.repose_calculation_from_images(self.path)
        # generate_video_from_path(path + '/repose/', 'plate_precision_repose.avi', fps)
        # generate_video_from_path(path, 'plate_precision.avi', fps)

        cv2.waitKey(0) & 0xFF == ord("q")
        cv2.destroyAllWindows()
        # cv2.imwrite('good2_repose.png', current_frame)

        # save_img_to_path(current_frame, 'good2_repose.png')
        self.data_json["kernel_size"]["value"] = self.kernel_size
        self.data_json["low_threshold"]["value"] = self.low_threshold
        self.data_json["high_threshold"]["value"] = self.high_threshold
        self.data_json["rho"]["value"] = self.rho
        self.data_json["theta"]["value"] = self.theta
        self.data_json["threshold"]["value"] = self.threshold
        self.data_json["min_line_length"]["value"] = self.min_line_length
        self.data_json["max_line_gap"]["value"] = self.max_line_gap
        self.data_json["current_image_mode"]["value"] = self.current_image_mode
        self.data_json["scale_percent"]["value"] = self.scale_percent
        self.data_json["line_visibility"]["value"] = self.line_visibility
        self.data_json["image_window_name"]["value"] = self.image_window_name
        self.data_json["trackbar_window_name"]["value"] = self.trackbar_window_name
        self.data_json["top_boundary"]["value"] = self.top_boundary
        self.data_json["bottom_boundary"]["value"] = self.bottom_boundary
        self.data_json["left_boundary"]["value"] = self.left_boundary
        self.data_json["right_boundary"]["value"] = self.right_boundary

        self.data_json["middle_line_position"]["value"] = self.middle_line_position
        self.data_json["hsv_hue_bottom"]["value"] = self.hsv_hue_bottom
        self.data_json["hsv_hue_top"]["value"] = self.hsv_hue_top
        self.data_json["hsv_saturation_bottom"]["value"] = self.hsv_saturation_bottom
        self.data_json["hsv_saturation_top"]["value"] = self.hsv_saturation_top
        self.data_json["hsv_value_bottom"]["value"] = self.hsv_value_bottom
        self.data_json["hsv_value_top"]["value"] = self.hsv_value_top

        print(self.middle_line_position)
        print(self.data_json)
        with open("config.json", "w") as ff:
            json.dump(self.data_json, ff, indent=4)


if __name__ == "__main__":
    app = AngleDetector()
    app.load_config()
    sys.exit(app.main())
