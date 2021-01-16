import colorsys
import random
import cv2
import os

GREEN = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA
BGR2RGB = cv2.COLOR_BGR2RGB
RGB2BGR = cv2.COLOR_RGB2BGR
BGR2HSV = cv2.COLOR_BGR2HSV
RGB2HSV = cv2.COLOR_RGB2HSV
HSV2RGB = cv2.COLOR_HSV2RGB
HSV2BGR = cv2.COLOR_HSV2BGR
BGR2GRAY = cv2.COLOR_BGR2GRAY
IMREAD_COLOR = cv2.IMREAD_COLOR
UPNP = cv2.SOLVEPNP_UPNP


class Camera(object):
    """Camera abstract class.
    By default this camera uses the openCV functionality.
    It can be inherited to overwrite methods in case another camera API exists.
    """
    def __init__(self, device_id=0, name='Camera'):
        # TODO load parameters from camera name. Use ``load`` method.
        self.device_id = device_id
        self.camera = None
        self.intrinsics = None
        self.distortion = None

    @property
    def intrinsics(self):
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, intrinsics):
        self._intrinsics = intrinsics

    @property
    def distortion(self):
        return self._distortion

    @distortion.setter
    def distortion(self, distortion):
        self._distortion = distortion

    def start(self):
        """ Starts capturing device
        """
        self.camera = cv2.VideoCapture(self.device_id)
        return self.camera

    def stop(self):
        """ Stops capturing device.
        """
        return self.camera.release()

    def read(self):
        """Reads camera input and returns a frame
        # Returns
            Array of camera
        """
        frame = self.camera.read()[1]
        return frame

    def is_open(self):
        """Checks if camera is open
        # Returns
            Boolean
        """
        return self.camera.isOpened()

    def calibrate(self):
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError


class VideoPlayer(object):
    """Performs visualization inferences in a real-time video.

    # Properties
        image_size: List of two integers. Output size of the displayed image.
        pipeline: Function. Should take BGR image as input and it should
            output a dictionary with key 'image' containing a visualization
            of the inferences.
            Built-in pipelines can be found in paz/processing/pipelines.py
    # Methods
        run()
        record()
    """

    def __init__(self, image_size, pipeline, camera):
        self.image_size = image_size
        self.pipeline = pipeline
        self.camera = camera

    def step(self):
        """ Runs the pipeline process once
        """
        if self.camera.is_open() is False:
            raise 'Camera has not started. Call ``start`` method.'

        frame = self.camera.read()
        if frame is None:
            print('Frame: None')
            return None
        return self.pipeline({'image': frame})

    def run(self):
        """Opens camera and starts continuous inference using ``pipeline``,
        until the user presses `q` inside the opened window.
        """
        self.camera.start()
        while True:
            results = self.step()
            image = cv2.resize(results['image'], tuple(self.image_size))
            cv2.imshow('webcam', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.camera.stop()
        cv2.destroyAllWindows()

    def record(self, name='video.avi', fps=20, fourCC='XVID'):
        """Opens camera and records continuous inference using ``pipeline``.
        # Arguments
            name: String. Video name. Must include the postfix .avi
            fps: Int. Frames per second
            fourCC: String. Indicates the four character code of the video.
            e.g. XVID, MJPG, X264
        """
        self.start()
        fourCC = cv2.VideoWriter_fourcc(*fourCC)
        writer = cv2.VideoWriter(name, fourCC, fps, self.image_size)
        while True:
            results = self.step()
            image = cv2.resize(results['image'], tuple(self.image_size))
            cv2.imshow('webcam', image)
            writer.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()
        writer.release()
        cv2.destroyAllWindows()


def cascade_classifier(path):
    """Cascade classifier with detectMultiScale() method for inference.
    # Arguments
        path: String. Path to default openCV XML format.
    """
    return cv2.CascadeClassifier(path)


def load_image(filepath, flags=cv2.IMREAD_COLOR):
    """Loads an image.
    # Arguments
        filepath: string with image path
        flags: Integers indicating flags about how to read image:
            1 or cv2.IMREAD_COLOR for BGR image.
            0 or cv2.IMREAD_GRAYSCALE for grayscale image.
           -1 or cv2.IMREAD_UNCHANGED for BGR with alpha-channel.
    # Returns
        Image as numpy array.
    """
    return cv2.imread(filepath, flags)


def resize_image(image, shape):
    """ Resizes image.
    # Arguments
        image: Numpy array.
        shape: List of two integer elements indicating new shape.
    """
    return cv2.resize(image, shape)


def save_image(filepath, image, *args):
    """Saves an image.
    # Arguments
        filepath: String with image path. It should include postfix e.g. .png
        image: Numpy array.
    """
    return cv2.imwrite(filepath, image, *args)


def save_images(save_path, images):
    """Saves multiple images in a directory
    # Arguments
        save_path: String. Path to directory. If path does not exist it will
        be created.
        images: List of numpy arrays.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for arg, image in enumerate(images):
        save_image(os.path.join(save_path, 'image_%03d.png' % arg), image)


def convert_image(image, flag):
    """Converts image to a different color space
    # Arguments
        image: Numpy array
        flag: OpenCV color flag e.g. cv2.COLOR_BGR2RGB or BGR2RGB
    """
    return cv2.cvtColor(image, flag)


def show_image(image, name='image', wait=True):
    """ Shows image in an external window.
    # Arguments
        image: Numpy array
        name: String indicating the window name.
    """
    cv2.imshow(name, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def draw_circle(image, point, color=GREEN, radius=5):
    """ Draws a cirle in image.
    # Arguments
        image: Numpy array of shape [H, W, 3].
        point: List/tuple of length two indicating (y,x) openCV coordinates.
        color: List of length three indicating BGR color of point.
        radius: Integer indicating the radius of the point to be drawn.
    """
    cv2.circle(image, tuple(point), radius, (0, 0, 0), cv2.FILLED)
    inner_radius = int(.8 * radius)
    cv2.circle(image, tuple(point), inner_radius, tuple(color), cv2.FILLED)


def put_text(image, text, point, scale, color, thickness):
    """Draws text in image.
    # Arguments
        image: Numpy array.
        text: String. Text to be drawn.
        point: Tuple of coordinates indicating the top corner of the text.
        scale: Float. Scale of text.
        color: Tuple of integers. BGR color coordinates.
        thickness: Integer. Thickness of the lines used for drawing text.
    """
    return cv2.putText(image, text, point, FONT, scale, color, thickness, LINE)


def draw_line(image, point_A, point_B, color=GREEN, thickness=5):
    """ Draws a line in image from point_A to point_B.
    # Arguments
        image: Numpy array of shape [H, W, 3].
        point_A: List/tuple of length two indicating (y,x) openCV coordinates.
        point_B: List/tuple of length two indicating (y,x) openCV coordinates.
        color: List of length three indicating BGR color of point.
        thickness: Integer indicating the thickness of the line to be drawn.
    """
    cv2.line(image, tuple(point_A), tuple(point_B), tuple(color), thickness)


def draw_rectangle(image, corner_A, corner_B, color, thickness):
    """ Draws a filled rectangle from corner_A to corner_B.
    # Arguments
        image: Numpy array of shape [H, W, 3].
        corner_A: List/tuple of length two indicating (y,x) openCV coordinates.
        corner_B: List/tuple of length two indicating (y,x) openCV coordinates.
        color: List of length three indicating BGR color of point.
        thickness: Integer/openCV Flag. Thickness of rectangle line.
            or for filled use cv2.FILLED flag.
    """
    return cv2.rectangle(
        image, tuple(corner_A), tuple(corner_B), tuple(color), thickness)


def draw_dot(image, point, color=GREEN, radius=5, filled=True):
    """ Draws a dot (small rectangle) in image.
    # Arguments
        image: Numpy array of shape [H, W, 3].
        point: List/tuple of length two indicating (y,x) openCV coordinates.
        color: List of length three indicating BGR color of point.
        radius: Integer indicating the radius of the point to be drawn.
        filled: Boolean. If `True` rectangle is filled with `color`.
    """
    # drawing outer black rectangle
    point_A = (point[0] - radius, point[1] - radius)
    point_B = (point[0] + radius, point[1] + radius)
    draw_rectangle(image, tuple(point_A), tuple(point_B), (0, 0, 0), filled)

    # drawing innner rectangle with given `color`
    inner_radius = int(.8 * radius)
    point_A = (point[0] - inner_radius, point[1] - inner_radius)
    point_B = (point[0] + inner_radius, point[1] + inner_radius)
    draw_rectangle(image, tuple(point_A), tuple(point_B), color, filled)


def draw_cube(image, points, color=GREEN, thickness=2):
    """ Draws a cube in image.
    # Arguments
        image: Numpy array of shape [H, W, 3].
        points: List of length 8  having each element a list
            of length two indicating (y,x) openCV coordinates.
        color: List of length three indicating BGR color of point.
        thickness: Integer indicating the thickness of the line to be drawn.
    """
    # draw bottom
    draw_line(image, points[0][0], points[1][0], color, thickness)
    draw_line(image, points[1][0], points[2][0], color, thickness)
    draw_line(image, points[3][0], points[2][0], color, thickness)
    draw_line(image, points[3][0], points[0][0], color, thickness)

    # draw top
    draw_line(image, points[4][0], points[5][0], color, thickness)
    draw_line(image, points[6][0], points[5][0], color, thickness)
    draw_line(image, points[6][0], points[7][0], color, thickness)
    draw_line(image, points[4][0], points[7][0], color, thickness)

    # draw sides
    draw_line(image, points[0][0], points[4][0], color, thickness)
    draw_line(image, points[7][0], points[3][0], color, thickness)
    draw_line(image, points[5][0], points[1][0], color, thickness)
    draw_line(image, points[2][0], points[6][0], color, thickness)

    # draw X mark on top
    draw_line(image, points[4][0], points[6][0], color, thickness)
    draw_line(image, points[5][0], points[7][0], color, thickness)

    # draw dots
    # [draw_dot(image, point, color, point_radii) for point in points]


def warp_affine(image, matrix, fill_color=[0, 0, 0]):
    """ Transforms `image` using an affine `matrix` transformation.
    # Arguments
        image: Numpy array.
        matrix: Numpy array of shape (2,3) indicating affine transformation.
        fill_color: List/tuple representing BGR use for filling empty space.
    """
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image, matrix, (width, height), borderValue=fill_color)


def draw_filled_polygon(image, vertices, color):
    """ Draws filled polygon
    # Arguments
        image: Numpy array.
        vertices: List of elements each having a list
            of length two indicating (y,x) openCV coordinates.
        color: Numpy array specifying BGR color of the polygon.
    """
    cv2.fillPoly(image, [vertices], color)


def median_blur(image, aperture):
    """ Blurs an image using the median filter.
    # Arguments
        image: Numpy array.
        aperture: Integer. Aperture linear size;
            it must be odd and greater than one.
    """
    return cv2.medianBlur(image, aperture)


def gaussian_blur(image, kernel_size):
    """ Blurs an image using the median filter.
    # Arguments
        image: Numpy array.
        kernel_size: List of two elements. Describes the gaussian kernel shape.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)


def lincolor(num_colors, saturation=1, value=1, normalized=False):
    """Creates a list of RGB colors linearly sampled from HSV space with
    randomised Saturation and Value

    # Arguments
        num_colors: Integer.
        saturation: Float or `None`. If float indicates saturation.
            If `None` it samples a random value.
        value: Float or `None`. If float indicates value.
            If `None` it samples a random value.

    # Returns
        List, for which each element contains a list with RGB color

    # References
        [Original implementation](https://github.com/jutanke/cselect)
    """
    RGB_colors = []
    hues = [value / num_colors for value in range(0, num_colors)]
    for hue in hues:

        if saturation is None:
            saturation = random.uniform(0.6, 1)

        if value is None:
            value = random.uniform(0.5, 1)

        RGB_color = colorsys.hsv_to_rgb(hue, saturation, value)
        if not normalized:
            RGB_color = [int(color * 255) for color in RGB_color]
        RGB_colors.append(RGB_color)
    return RGB_colors


def solve_PNP(points3D, points2D, camera, solver):
    """Calculates 6D pose from 3D points and 2D keypoints correspondences.
    # Arguments
        points: Numpy array of shape (num_points, 3).
            Model 3D points known in advance.
        keypoints: Numpy array of shape (num_points, 2).
            Predicted 2D keypoints of object
        camera intrinsics: Numpy array of shape (3, 3) calculated from
        the openCV calibrateCamera function
        solver: Flag from e.g openCV.SOLVEPNP_UPNP
        distortion: Numpy array of shape of 5 elements calculated from
        the openCV calibrateCamera function

    # Returns
        A list containing success flag, rotation and translation components
        of the 6D pose.

    # References
        https://docs.opencv.org/2.4/modules/calib3d/doc/calib3d.html
    """
    return cv2.solvePnPRansac(points3D, points2D, camera.intrinsics, None,
                        flags=cv2.SOLVEPNP_EPNP,reprojectionError=5,iterationsCount=100)
    # return cv2.solvePnP(points3D, points2D, camera.intrinsics,
                        # camera.distortion, None, None, False, solver)


def project_points3D(points3D, pose6D, camera):
    point2D, jacobian = cv2.projectPoints(
        points3D, pose6D.rotation_vector, pose6D.translation,
        camera.intrinsics, camera.distortion)
    return point2D
