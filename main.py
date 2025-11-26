from pydantic import BaseModel
import cv2
import math
import colorsys
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from panoptic.core.plugin.plugin import APlugin
from panoptic.models import ActionContext, PropertyType, PropertyMode, DbCommit, Instance, ImageProperty, Property
from panoptic.models.results import ActionResult, Notif, NotifType, Group, Score, ScoreList
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface


class ColorsPlugin(APlugin):
    def __init__(self, project: PluginProjectInterface, plugin_path: str, name: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.add_action_easy(self.compute_colors, ['execute'])
        self.add_action_easy(self.cluster_by_colors, ['group'])

    async def compute_colors(self, context: ActionContext):
        instances = await self.project.get_instances(ids=context.instance_ids)
        uniques = list({i.sha1: i for i in instances}.values())
        res = {}

        for i in uniques:
            # Utiliser run_async pour éviter de bloquer le thread principal
            try:
                rgb_array = await self.project.run_async(get_main_color, i.url)
                values = step(rgb_array, 8)
                res[i.sha1] = values
            except:
                print("error for image ", i.url) 

        await self.save_values(res)

        # Retourner un ActionResult avec une notification de succès
        notif = Notif(
            type=NotifType.INFO,
            name='compute_colors',
            message=f'Couleurs calculées pour {len(uniques)} images'
        )
        return ActionResult(notifs=[notif])

    async def cluster_by_colors(self, context: ActionContext, nb_clusters: int = 5, color_space: str = "RGB"):
        """
        Crée des clusters d'images basés sur leurs couleurs dominantes.

        :param context: Contexte de l'action contenant les IDs des instances
        :param nb_clusters: Nombre de clusters à créer
        :param color_space: Espace colorimétrique à utiliser ('RGB', 'HSV', ou 'ALL')
        """
        # Récupérer les instances
        instances = await self.project.get_instances(ids=context.instance_ids)

        # Récupérer les propriétés de couleur
        all_properties = await self.project.get_properties()
        color_props = {p.name: p for p in all_properties if p.name.startswith('color_')}

        # Vérifier que les propriétés de couleur existent
        if not color_props:
            notif = Notif(
                type=NotifType.ERROR,
                name='cluster_by_colors',
                message='Aucune propriété de couleur trouvée. Veuillez d\'abord calculer les couleurs.'
            )
            return ActionResult(notifs=[notif])

        # Récupérer les valeurs de propriétés pour les instances
        instance_ids = [i.id for i in instances]
        property_ids = [p.id for p in color_props.values()]
        values = await self.project.get_image_property_values(
            property_ids=property_ids,
            sha1s=[i.sha1 for i in instances]
        )

        # Organiser les données : sha1 -> {property_name: value}
        sha1_to_colors = {}
        for value in values:
            sha1 = value.sha1
            prop_name = next((name for name, p in color_props.items() if p.id == value.property_id), None)
            if sha1 not in sha1_to_colors:
                sha1_to_colors[sha1] = {}
            if prop_name:
                sha1_to_colors[sha1][prop_name] = value.value

        # Filtrer les instances qui ont des valeurs de couleur
        instances_with_colors = [i for i in instances if i.sha1 in sha1_to_colors and len(sha1_to_colors[i.sha1]) >= 3]

        if len(instances_with_colors) < nb_clusters:
            notif = Notif(
                type=NotifType.WARNING,
                name='cluster_by_colors',
                message=f'Pas assez d\'images avec des couleurs ({len(instances_with_colors)}) pour créer {nb_clusters} clusters.'
            )
            return ActionResult(notifs=[notif])

        # Préparer les vecteurs de features selon l'espace colorimétrique
        feature_vectors = []
        for instance in instances_with_colors:
            colors = sha1_to_colors[instance.sha1]

            if color_space == "RGB":
                features = [
                    colors.get('color_R', 0),
                    colors.get('color_G', 0),
                    colors.get('color_B', 0)
                ]
            elif color_space == "HSV":
                features = [
                    colors.get('color_H', 0),
                    colors.get('color_S', 0),
                    colors.get('color_V', 0)
                ]
            else:  # ALL
                features = [
                    colors.get('color_R', 0),
                    colors.get('color_G', 0),
                    colors.get('color_B', 0),
                    colors.get('color_H', 0),
                    colors.get('color_S', 0),
                    colors.get('color_V', 0),
                    colors.get('color_L', 0)
                ]

            feature_vectors.append(features)

        # Effectuer le clustering
        feature_array = np.array(feature_vectors)
        cluster_labels, cluster_centers = await self.project.run_async(
            perform_kmeans_clustering,
            feature_array,
            nb_clusters
        )

        # Organiser les résultats par cluster
        cluster_groups = {}
        for idx, label in enumerate(cluster_labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            instance = instances_with_colors[idx]
            cluster_groups[label].append({
                'id': instance.id,
                'distance': np.linalg.norm(feature_array[idx] - cluster_centers[label])
            })

        # Créer les objets Group pour l'interface
        groups = []
        for cluster_id in sorted(cluster_groups.keys()):
            cluster_items = cluster_groups[cluster_id]

            # Trier par distance au centre (les plus proches d'abord)
            cluster_items.sort(key=lambda x: x['distance'])

            ids = [item['id'] for item in cluster_items]

            # Obtenir les couleurs moyennes du cluster
            center = cluster_centers[cluster_id]
            if color_space == "RGB":
                color_desc = f"RGB({int(center[0])}, {int(center[1])}, {int(center[2])})"
            elif color_space == "HSV":
                color_desc = f"HSV({int(center[0])}, {int(center[1])}, {int(center[2])})"
            else:
                color_desc = f"RGB({int(center[0])}, {int(center[1])}, {int(center[2])})"

            group = Group(
                ids=ids,
                name=f"Cluster {cluster_id + 1}: {color_desc}",
            )
            groups.append(group)

        return ActionResult(groups=groups)

    async def save_values(self, colors):
        commit = DbCommit()

        # Créer les propriétés
        properties = []
        for letter in ['R', 'G', 'B', 'H', 'S', 'V', 'L']:
            prop = await self.project.get_or_create_property(
                "color_" + letter,
                PropertyType.number,
                PropertyMode.sha1
            )
            properties.append(prop)

        commit.properties.extend(properties)

        # Ajouter les valeurs pour chaque image
        for sha1 in colors:
            for index, prop in enumerate(properties):
                commit.image_values.append(
                    ImageProperty(
                        property_id=prop.id,
                        sha1=sha1,
                        value=colors[sha1][index]
                    )
                )

        # Utiliser do() pour permettre l'annulation/rétablissement
        await self.project.do(commit)


def get_main_color(image_path, n_colors=1):

    pil_image = Image.open(image_path)
    # Convertir PIL en array numpy
    image = np.array(pil_image)
    # Convertir RGB -> BGR si nécessaire
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Vérifier que l'image est valide
    if image is None or image.size == 0:
        print(f"Erreur: impossible de charger l'image {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Redimensionner l'image pour accélérer le traitement
    resized_image = cv2.resize(image_rgb, (100, 100), interpolation=cv2.INTER_AREA)
    # Reshape pour préparer l'image au clustering
    pixels = resized_image.reshape(-1, 3)
    # Utilisation de KMeans pour trouver la couleur dominante
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(pixels)
    # Récupérer la couleur centrale (la plus représentée)
    main_color = kmeans.cluster_centers_[0].astype(int)
    return main_color


def perform_kmeans_clustering(feature_array, n_clusters):
    """
    Effectue le clustering KMeans sur les features de couleur.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(feature_array)
    centers = kmeans.cluster_centers_
    return labels, centers


def rgb_to_hsl(rgb_arr):
    # Extraire les valeurs R, G, B
    R, G, B = rgb_arr
    # Convertir en HSV (OpenCV attend l'image au format BGR)
    main_color_bgr = np.uint8([[rgb_arr[::-1]]])  # Convertir en BGR pour OpenCV
    hsv_color = cv2.cvtColor(main_color_bgr, cv2.COLOR_BGR2HSV)[0][0]

    # Extraire les valeurs H, S, V
    H, S, V = hsv_color

    # Retourner les valeurs sous forme de dictionnaire
    return [round(R), round(G), round(B), int(H), int(S), int(V)]


def step(rgb_arr, repetitions=1):
    """
    taken from https://www.alanzucconi.com/2015/09/30/colour-sorting/
    """
    r, g, b = rgb_arr
    r_norm, g_norm, b_norm = r / 255, g / 255, b / 255
    l = math.sqrt(.241 * r + .691 * g + .068 * b)
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    h2 = int(h * repetitions)
    l2 = int(l * repetitions)
    v2 = int(v * repetitions)
    s2 = int(s * repetitions)
    return [int(r), int(g), int(b), h2, v2, s2, l2]
