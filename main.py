import colorsys
import io
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import msgspec
import numpy as np
from PIL import Image

from panoptic.core.databases.data.models import DataCommit, Property, Sha1Value
from panoptic.core.databases.media.models import Map
from panoptic.core.plugin.plugin import APlugin
from panoptic.core.task.task import Task
from panoptic.models.action_models import (
    ActionContext, ActionResult, Group, Notif, NotifType, ScoreList,
)

PROPERTY_PREFIX = 'color_'
PROPERTY_GROUP = 'Colors'
BATCH_SIZE = 500
IO_WORKERS = 8


class ColorComponent(Enum):
    hue = 'Hue'
    saturation = 'Saturation'
    value = 'Value'
    luminance = 'Luminance'
    red = 'Red'
    green = 'Green'
    blue = 'Blue'


class ColorSpace(Enum):
    rgb = 'RGB'
    hsv = 'HSV'
    all = 'ALL'


# component -> (property letter, min, max)
COMPONENTS = {
    ColorComponent.red: ('R', 0, 255),
    ColorComponent.green: ('G', 0, 255),
    ColorComponent.blue: ('B', 0, 255),
    ColorComponent.hue: ('H', 0, 360),
    ColorComponent.saturation: ('S', 0, 100),
    ColorComponent.value: ('V', 0, 100),
    ColorComponent.luminance: ('L', 0, 100),
}
LETTERS = [letter for letter, _, _ in COMPONENTS.values()]


class ColorsPlugin(APlugin):
    """
    Computes the mean color of images (RGB, HSV and perceived luminance) and stores it in
    color_* properties, to filter, sort, group or lay out images by color.
    """

    def __init__(self, name: str, project, plugin_path: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self._props_lock = threading.Lock()
        self.add_action_easy(self.compute_colors, ['execute'])
        self.add_action_easy(self.cluster_by_colors, ['group'])
        self.add_action_easy(self.color_map, ['map', 'execute'])

    def compute_colors(self, context: ActionContext) -> ActionResult:
        """Compute the mean color of the selected images and store it in color_* properties."""
        sha1s = self._get_sha1s(context)
        if sha1s:
            self.project.add_task(ComputeColorsTask(self, sha1s))
        return ActionResult(notifs=[Notif(
            NotifType.INFO, name='compute_colors',
            message=f'Started computing colors for {len(sha1s)} images',
        )])

    def cluster_by_colors(self, context: ActionContext, nb_clusters: int = 5,
                          color_space: ColorSpace = ColorSpace.rgb) -> ActionResult:
        """Cluster images by their mean color with K-means.
        @nb_clusters: number of clusters to create
        @color_space: color space the distances are computed in
        """
        colors = self._ensure_colors(self._get_sha1s(context))
        if len(colors) < nb_clusters:
            return ActionResult(notifs=[Notif(
                NotifType.WARNING, name='cluster_by_colors',
                message=f'Not enough images with colors ({len(colors)}) to create {nb_clusters} clusters.',
            )])

        sha1s = list(colors.keys())
        features = np.array([to_features(colors[s], color_space) for s in sha1s])
        labels, centers = self.project.run_in_executor(kmeans, features, nb_clusters)
        distances = np.linalg.norm(features - centers[labels], axis=1)

        groups = []
        for cluster in range(nb_clusters):
            members = sorted(np.flatnonzero(labels == cluster), key=lambda i: distances[i])
            if not members:
                continue
            rgb = np.mean([[colors[sha1s[i]][c] for c in 'RGB'] for i in members], axis=0)
            groups.append(Group(
                sha1s=[sha1s[i] for i in members],
                scores=ScoreList(values=[float(distances[i]) for i in members],
                                 min=0, max=float(distances.max()) or 1, max_is_best=False,
                                 description='Distance to the cluster center'),
                name=f'Cluster {len(groups) + 1}: {rgb_to_hex(rgb)}',
            ))
        return ActionResult(groups=groups)

    def color_map(self, context: ActionContext, x_axis: ColorComponent = ColorComponent.hue,
                  y_axis: ColorComponent = ColorComponent.saturation, radial: bool = False,
                  reverse_x: bool = False, reverse_y: bool = False, map_name: str = '') -> ActionResult:
        """Create a spatial view placing images by two color components.
        @x_axis: color component on the horizontal axis (the angle in radial layout)
        @y_axis: color component on the vertical axis (the distance to the center in radial layout)
        @radial: radial layout, e.g. a color wheel with Hue and Saturation
        @reverse_x: reverse the horizontal axis (the direction of rotation in radial layout)
        @reverse_y: reverse the vertical axis (highest values at the center in radial layout)
        @map_name: name for the saved map (auto-generated if empty)
        """
        colors = self._ensure_colors(self._get_sha1s(context))
        if not colors:
            return ActionResult(notifs=[Notif(
                NotifType.ERROR, name='color_map', message='No image color could be computed',
            )])

        position = radial_position if radial else cartesian_position
        flat = []
        for sha1, values in colors.items():
            u = normalize(values, x_axis, reverse_x)
            v = normalize(values, y_axis, reverse_y)
            flat += [sha1, *position(u, v)]
        default_name = f'color{" radial" if radial else ""}: {x_axis.value} / {y_axis.value}'
        point_map = self.project.upsert_map(Map(
            id=-1, source=self.name, name=map_name or default_name,
            key='sha1', count=len(colors), data=flat,
        ))
        return ActionResult(value=msgspec.structs.asdict(point_map))

    def compute_and_save(self, sha1s: list[str]) -> dict[str, dict]:
        colors = compute_colors(self.project, sha1s)
        self._save_colors(colors)
        return colors

    def _ensure_colors(self, sha1s: list[str]) -> dict[str, dict]:
        """Stored colors of `sha1s`, computing the missing ones."""
        colors = self._read_colors(sha1s)
        missing = [s for s in sha1s if s not in colors]
        if missing:
            colors.update(self.compute_and_save(missing))
        return colors

    def _read_colors(self, sha1s: list[str]) -> dict[str, dict]:
        props = self._get_properties(create=False)
        if len(props) < len(LETTERS) or not sha1s:
            return {}
        letter_by_id = {p.id: letter for letter, p in props.items()}
        values = self.project.get_sha1_values(property_id=list(letter_by_id), sha1=sha1s)
        colors: dict[str, dict] = {}
        for v in values:
            if v.value is not None:
                colors.setdefault(v.sha1, {})[letter_by_id[v.property_id]] = v.value
        return {s: c for s, c in colors.items() if len(c) == len(LETTERS)}

    def _save_colors(self, colors: dict[str, dict]) -> None:
        if not colors:
            return
        props = self._get_properties(create=True)
        commit = DataCommit(sha1_values=[
            Sha1Value(property_id=props[letter].id, sha1=sha1, value=values[letter])
            for sha1, values in colors.items() for letter in LETTERS
        ])
        self.project.apply_commit(commit)

    def _get_properties(self, create: bool) -> dict[str, Property]:
        with self._props_lock:
            existing = {p.name: p for p in self.project.get_properties()
                        if p.dtype == 'number' and p.mode == 'sha1'}
            props = {letter: existing[PROPERTY_PREFIX + letter] for letter in LETTERS
                     if PROPERTY_PREFIX + letter in existing}
            if not create:
                return props

            group_id = self._get_group_id()
            ungrouped = [msgspec.structs.replace(p, property_group_id=group_id)
                         for p in props.values() if p.property_group_id is None]
            if group_id is not None and ungrouped:
                self.project.apply_commit(DataCommit(properties=ungrouped))
            for letter in LETTERS:
                if letter not in props:
                    props[letter] = self.project.add_property(
                        PROPERTY_PREFIX + letter, 'number', 'sha1',
                        readonly=True, property_group_id=group_id,
                    )
            return props

    def _get_group_id(self) -> int | None:
        # older Panoptic versions don't let plugins manage property groups
        if not hasattr(self.project, 'add_property_group'):
            return None
        group = next((g for g in self.project.get_property_groups() if g.name == PROPERTY_GROUP), None)
        return (group or self.project.add_property_group(PROPERTY_GROUP)).id

    def _get_sha1s(self, context: ActionContext) -> list[str]:
        if context.instance_ids:
            instances = self.project.get_instances(id=context.instance_ids)
        else:
            instances = self.project.get_instances()
        return list(dict.fromkeys(i.sha1 for i in instances if i.sha1))


class ComputeColorsTask(Task):
    def __init__(self, plugin: ColorsPlugin, sha1s: list[str]):
        super().__init__()
        self.plugin = plugin
        self.sha1s = sha1s
        self.name = 'Colors'

    def start(self) -> None:
        self.state.total = len(self.sha1s)
        self._notify()
        for i in range(0, len(self.sha1s), BATCH_SIZE):
            if self.is_cancelled():
                break
            chunk = self.sha1s[i:i + BATCH_SIZE]
            try:
                colors = self.plugin.compute_and_save(chunk)
            except Exception as e:
                print(f'PanopticColor: failed to compute colors: {e!r}')
                self.state.failed += len(chunk)
                self._notify()
                continue
            self.state.done += len(colors)
            self.state.failed += len(chunk) - len(colors)
            self._notify()


def compute_colors(project, sha1s: list[str]) -> dict[str, dict]:
    """Color values of each sha1, computed from its smallest stored rendition."""
    # The plugin interface has no public image access yet, hence _media_db().
    with project._media_db() as db:
        image_type = smallest_image_type(db.get_image_types())
        if image_type is None:
            return {}
        images = db.get_images(type_id=image_type, sha1=sha1s)

    def compute(image):
        try:
            return image.sha1, color_values(mean_rgb(image.data))
        except Exception as e:
            print(f'PanopticColor: cannot read image {image.sha1}: {e}')
            return None

    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        return dict(r for r in pool.map(compute, images) if r is not None)


def smallest_image_type(image_types) -> int | None:
    if not image_types:
        return None
    return min(image_types, key=lambda t: max(t.width or math.inf, t.height or math.inf)).id


def mean_rgb(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert('RGB')
    return np.asarray(image, dtype=np.float64).reshape(-1, 3).mean(axis=0)


def color_values(rgb) -> dict[str, int]:
    r, g, b = (float(c) for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    # perceived brightness, see https://www.alanzucconi.com/2015/09/30/colour-sorting/
    lum = math.sqrt(.241 * r ** 2 + .691 * g ** 2 + .068 * b ** 2) / 255
    return {
        'R': round(r), 'G': round(g), 'B': round(b),
        'H': round(h * 360) % 360, 'S': round(s * 100), 'V': round(v * 100), 'L': round(lum * 100),
    }


def to_features(values: dict, color_space: ColorSpace) -> list[float]:
    rgb = [values[c] / 255 * 100 for c in 'RGB']
    # hue is circular: place colors in the HSV cone so that 359° is next to 0°
    angle = math.radians(values['H'])
    hsv = [values['S'] * math.cos(angle), values['S'] * math.sin(angle), values['V']]
    if color_space == ColorSpace.rgb:
        return rgb
    if color_space == ColorSpace.hsv:
        return hsv
    return rgb + hsv + [values['L']]


def normalize(values: dict, component: ColorComponent, reverse: bool = False) -> float:
    """Component value mapped to [0, 1]."""
    letter, lo, hi = COMPONENTS[component]
    n = (values[letter] - lo) / (hi - lo)
    return 1 - n if reverse else n


# Panoptic maps fit in a disc of radius 100
def cartesian_position(u: float, v: float) -> tuple[float, float]:
    return u * 200 - 100, v * 200 - 100


def radial_position(u: float, v: float) -> tuple[float, float]:
    angle = u * 2 * math.pi
    return v * 100 * math.cos(angle), v * 100 * math.sin(angle)


def rgb_to_hex(rgb) -> str:
    return '#' + ''.join(f'{int(round(c)):02x}' for c in rgb)


def kmeans(features: np.ndarray, n_clusters: int):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(features)
    return labels, model.cluster_centers_
