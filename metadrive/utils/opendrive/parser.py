# -*- coding: utf-8 -*-
"""
These scripts are copied from https://github.com/liuyf5231/opendriveparser. Credit: https://github.com/liuyf5231
"""
import numpy as np
from lxml import etree

import vqa.vqagen.utils.qa_utils
from metadrive.utils.opendrive.elements.opendrive import OpenDrive, Header
from metadrive.utils.opendrive.elements.road import Road
from metadrive.utils.opendrive.elements.roadLink import (
    Predecessor as RoadLinkPredecessor,
    Successor as RoadLinkSuccessor,
    Neighbor as RoadLinkNeighbor,
)
from metadrive.utils.opendrive.elements.roadtype import (
    RoadType,
    Speed as RoadTypeSpeed,
)
from metadrive.utils.opendrive.elements.roadElevationProfile import (
    ElevationRecord as RoadElevationProfile,
)
from metadrive.utils.opendrive.elements.roadLateralProfile import (
    Superelevation as RoadLateralProfileSuperelevation,
    Crossfall as RoadLateralProfileCrossfall,
    Shape as RoadLateralProfileShape,
)
from metadrive.utils.opendrive.elements.roadLanes import (
    LaneOffset as RoadLanesLaneOffset,
    Lane as RoadLaneSectionLane,
    LaneSection as RoadLanesSection,
    LaneWidth as RoadLaneSectionLaneWidth,
    LaneBorder as RoadLaneSectionLaneBorder,
)
from metadrive.utils.opendrive.elements.junction import (
    Junction,
    Connection as JunctionConnection,
    LaneLink as JunctionConnectionLaneLink,
)

__author__ = "Benjamin Orthen, Stefan Urban"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["Priority Program SPP 1835 Cooperative Interacting Automobiles"]
__version__ = "1.0.2"
__maintainer__ = "Benjamin Orthen"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Released"


def parse_opendrive(root_node) -> OpenDrive:
    """Tries to parse XML tree, returns OpenDRIVE object

    Args:
      root_node:

    Returns:
      The object representing an OpenDrive specification.

    """

    # Only accept lxml element
    if not etree.iselement(root_node):
        raise TypeError("Argument root_node is not a xml element")

    opendrive = OpenDrive()

    # Header
    header = root_node.find("header")
    if header is not None:
        parse_opendrive_header(opendrive, header)

    # Junctions
    for junction in root_node.findall("junction"):
        parse_opendrive_junction(opendrive, junction)

    # Load roads
    for road in root_node.findall("road"):
        parse_opendrive_road(opendrive, road)

    return opendrive


def parse_opendrive_road_link(newRoad, opendrive_road_link):
    """

    Args:
      newRoad:
      opendrive_road_link:

    """
    predecessor = opendrive_road_link.find("predecessor")

    if predecessor is not None:
        newRoad.link.predecessor = RoadLinkPredecessor(
            vqa.vqagen.utils.qa_utils.get("elementType"),
            vqa.vqagen.utils.qa_utils.get("elementId"),
            vqa.vqagen.utils.qa_utils.get("contactPoint"),
        )

    successor = opendrive_road_link.find("successor")

    if successor is not None:
        newRoad.link.successor = RoadLinkSuccessor(
            vqa.vqagen.utils.qa_utils.get("elementType"),
            vqa.vqagen.utils.qa_utils.get("elementId"),
            vqa.vqagen.utils.qa_utils.get("contactPoint"),
        )

    for neighbor in opendrive_road_link.findall("neighbor"):
        newNeighbor = RoadLinkNeighbor(vqa.vqagen.utils.qa_utils.get("side"), vqa.vqagen.utils.qa_utils.get("elementId"), vqa.vqagen.utils.qa_utils.get("direction"))

        newRoad.link.neighbors.append(newNeighbor)


def parse_opendrive_road_type(road, opendrive_xml_road_type: etree.ElementTree):
    """Parse opendrive road type and append to road object.

    Args:
      road: Road to append the parsed road_type to types.
      opendrive_xml_road_type: XML element which contains the information.
      opendrive_xml_road_type: etree.ElementTree:

    """
    speed = None
    if opendrive_xml_road_type.find("speed") is not None:
        speed = RoadTypeSpeed(
            max_speed=vqa.vqagen.utils.qa_utils.get("max"),
            unit=vqa.vqagen.utils.qa_utils.get("unit"),
        )

    road_type = RoadType(
        s_pos=vqa.vqagen.utils.qa_utils.get("s"),
        use_type=vqa.vqagen.utils.qa_utils.get("type"),
        speed=speed,
    )
    road.types.append(road_type)


def parse_opendrive_road_geometry(newRoad, road_geometry):
    """

    Args:
      newRoad:
      road_geometry:

    """

    startCoord = [float(vqa.vqagen.utils.qa_utils.get("x")), float(vqa.vqagen.utils.qa_utils.get("y"))]

    if road_geometry.find("line") is not None:
        newRoad.planView.addLine(
            startCoord,
            float(vqa.vqagen.utils.qa_utils.get("hdg")),
            float(vqa.vqagen.utils.qa_utils.get("length")),
        )

    elif road_geometry.find("spiral") is not None:
        newRoad.planView.addSpiral(
            startCoord,
            float(vqa.vqagen.utils.qa_utils.get("hdg")),
            float(vqa.vqagen.utils.qa_utils.get("length")),
            float(vqa.vqagen.utils.qa_utils.get("curvStart")),
            float(vqa.vqagen.utils.qa_utils.get("curvEnd")),
        )

    elif road_geometry.find("arc") is not None:
        newRoad.planView.addArc(
            startCoord,
            float(vqa.vqagen.utils.qa_utils.get("hdg")),
            float(vqa.vqagen.utils.qa_utils.get("length")),
            float(vqa.vqagen.utils.qa_utils.get("curvature")),
        )

    elif road_geometry.find("poly3") is not None:
        newRoad.planView.addPoly3(
            startCoord,
            float(vqa.vqagen.utils.qa_utils.get("hdg")),
            float(vqa.vqagen.utils.qa_utils.get("length")),
            float(vqa.vqagen.utils.qa_utils.get("a")),
            float(vqa.vqagen.utils.qa_utils.get("b")),
            float(vqa.vqagen.utils.qa_utils.get("c")),
            float(vqa.vqagen.utils.qa_utils.get("d")),
        )
        # raise NotImplementedError()

    elif road_geometry.find("paramPoly3") is not None:
        if vqa.vqagen.utils.qa_utils.get("pRange"):

            if vqa.vqagen.utils.qa_utils.get("pRange") == "arcLength":
                pMax = float(vqa.vqagen.utils.qa_utils.get("length"))
            else:
                pMax = None
        else:
            pMax = None

        newRoad.planView.addParamPoly3(
            startCoord,
            float(vqa.vqagen.utils.qa_utils.get("hdg")),
            float(vqa.vqagen.utils.qa_utils.get("length")),
            float(vqa.vqagen.utils.qa_utils.get("aU")),
            float(vqa.vqagen.utils.qa_utils.get("bU")),
            float(vqa.vqagen.utils.qa_utils.get("cU")),
            float(vqa.vqagen.utils.qa_utils.get("dU")),
            float(vqa.vqagen.utils.qa_utils.get("aV")),
            float(vqa.vqagen.utils.qa_utils.get("bV")),
            float(vqa.vqagen.utils.qa_utils.get("cV")),
            float(vqa.vqagen.utils.qa_utils.get("dV")),
            pMax,
        )

    else:
        raise Exception("invalid xml")


def parse_opendrive_road_elevation_profile(newRoad, road_elevation_profile):
    """

    Args:
      newRoad:
      road_elevation_profile:

    """

    for elevation in road_elevation_profile.findall("elevation"):
        newElevation = (
            RoadElevationProfile(
                float(vqa.vqagen.utils.qa_utils.get("a")),
                float(vqa.vqagen.utils.qa_utils.get("b")),
                float(vqa.vqagen.utils.qa_utils.get("c")),
                float(vqa.vqagen.utils.qa_utils.get("d")),
                start_pos=float(vqa.vqagen.utils.qa_utils.get("s")),
            ),
        )

        newRoad.elevationProfile.elevations.append(newElevation)


def parse_opendrive_road_lateral_profile(newRoad, road_lateral_profile):
    """

    Args:
      newRoad:
      road_lateral_profile:

    """

    for superelevation in road_lateral_profile.findall("superelevation"):
        newSuperelevation = RoadLateralProfileSuperelevation(
            float(vqa.vqagen.utils.qa_utils.get("a")),
            float(vqa.vqagen.utils.qa_utils.get("b")),
            float(vqa.vqagen.utils.qa_utils.get("c")),
            float(vqa.vqagen.utils.qa_utils.get("d")),
            start_pos=float(vqa.vqagen.utils.qa_utils.get("s")),
        )

        newRoad.lateralProfile.superelevations.append(newSuperelevation)

    for crossfall in road_lateral_profile.findall("crossfall"):
        newCrossfall = RoadLateralProfileCrossfall(
            float(vqa.vqagen.utils.qa_utils.get("a")),
            float(vqa.vqagen.utils.qa_utils.get("b")),
            float(vqa.vqagen.utils.qa_utils.get("c")),
            float(vqa.vqagen.utils.qa_utils.get("d")),
            side=vqa.vqagen.utils.qa_utils.get("side"),
            start_pos=float(vqa.vqagen.utils.qa_utils.get("s")),
        )

        newRoad.lateralProfile.crossfalls.append(newCrossfall)

    for shape in road_lateral_profile.findall("shape"):
        newShape = RoadLateralProfileShape(
            float(vqa.vqagen.utils.qa_utils.get("a")),
            float(vqa.vqagen.utils.qa_utils.get("b")),
            float(vqa.vqagen.utils.qa_utils.get("c")),
            float(vqa.vqagen.utils.qa_utils.get("d")),
            start_pos=float(vqa.vqagen.utils.qa_utils.get("s")),
            start_pos_t=float(vqa.vqagen.utils.qa_utils.get("t")),
        )

        newRoad.lateralProfile.shapes.append(newShape)


def parse_opendrive_road_lane_offset(newRoad, lane_offset):
    """

    Args:
      newRoad:
      lane_offset:

    """

    newLaneOffset = RoadLanesLaneOffset(
        float(vqa.vqagen.utils.qa_utils.get("a")),
        float(vqa.vqagen.utils.qa_utils.get("b")),
        float(vqa.vqagen.utils.qa_utils.get("c")),
        float(vqa.vqagen.utils.qa_utils.get("d")),
        start_pos=float(vqa.vqagen.utils.qa_utils.get("s")),
    )

    newRoad.lanes.laneOffsets.append(newLaneOffset)


def parse_opendrive_road_lane_section(newRoad, lane_section_id, lane_section):
    """

    Args:
      newRoad:
      lane_section_id:
      lane_section:

    """

    newLaneSection = RoadLanesSection(road=newRoad)

    # Manually enumerate lane sections for referencing purposes
    newLaneSection.idx = lane_section_id

    newLaneSection.sPos = float(vqa.vqagen.utils.qa_utils.get("s"))
    newLaneSection.singleSide = vqa.vqagen.utils.qa_utils.get("singleSide")

    sides = dict(
        left=newLaneSection.leftLanes,
        center=newLaneSection.centerLanes,
        right=newLaneSection.rightLanes,
    )

    for sideTag, newSideLanes in sides.items():

        side = lane_section.find(sideTag)

        # It is possible one side is not present
        if side is None:
            continue

        for lane in side.findall("lane"):

            new_lane = RoadLaneSectionLane(parentRoad=newRoad, lane_section=newLaneSection)
            new_lane.id = vqa.vqagen.utils.qa_utils.get("id")
            new_lane.type = vqa.vqagen.utils.qa_utils.get("type")

            # In some sample files the level is not specified according to the OpenDRIVE spec
            new_lane.level = ("true" if vqa.vqagen.utils.qa_utils.get("level") in [1, "1", "true"] else "false")

            # Lane Links
            if lane.find("link") is not None:

                if lane.find("link").find("predecessor") is not None:
                    new_lane.link.predecessorId = (vqa.vqagen.utils.qa_utils.get("id"))

                if lane.find("link").find("successor") is not None:
                    new_lane.link.successorId = (vqa.vqagen.utils.qa_utils.get("id"))

            # Width
            for widthIdx, width in enumerate(lane.findall("width")):
                newWidth = RoadLaneSectionLaneWidth(
                    float(vqa.vqagen.utils.qa_utils.get("a")),
                    float(vqa.vqagen.utils.qa_utils.get("b")),
                    float(vqa.vqagen.utils.qa_utils.get("c")),
                    float(vqa.vqagen.utils.qa_utils.get("d")),
                    idx=widthIdx,
                    start_offset=float(vqa.vqagen.utils.qa_utils.get("sOffset")),
                )

                new_lane.widths.append(newWidth)

            # Border
            for borderIdx, border in enumerate(lane.findall("border")):
                newBorder = RoadLaneSectionLaneBorder(
                    float(vqa.vqagen.utils.qa_utils.get("a")),
                    float(vqa.vqagen.utils.qa_utils.get("b")),
                    float(vqa.vqagen.utils.qa_utils.get("c")),
                    float(vqa.vqagen.utils.qa_utils.get("d")),
                    idx=borderIdx,
                    start_offset=float(vqa.vqagen.utils.qa_utils.get("sOffset")),
                )

                new_lane.borders.append(newBorder)

            if lane.find("width") is None and lane.find("border") is not None:
                new_lane.widths = new_lane.borders
                new_lane.has_border_record = True

            # Road Marks
            # TODO implementation
            if lane.find("roadMark") is not None:
                new_lane.roadMark = dict(lane.find("roadMark").attrib)
            else:
                new_lane.roadMark = None
                # new_lane.has_border_record = True

            # Material
            # TODO implementation

            # Visiblility
            # TODO implementation

            # Speed
            # TODO implementation

            # Access
            # TODO implementation

            # Lane Height
            # TODO implementation

            # Rules
            # TODO implementation

            newSideLanes.append(new_lane)

    newRoad.lanes.lane_sections.append(newLaneSection)


def parse_opendrive_road(opendrive, road):
    """

    Args:
      opendrive:
      road:

    """

    newRoad = Road()

    newRoad.id = int(vqa.vqagen.utils.qa_utils.get("id"))
    newRoad.name = vqa.vqagen.utils.qa_utils.get("name")

    junctionId = int(vqa.vqagen.utils.qa_utils.get("junction")) if vqa.vqagen.utils.qa_utils.get("junction") != "-1" else None

    if junctionId:
        newRoad.junction = opendrive.getJunction(junctionId)

    # TODO verify road length
    newRoad.length = float(vqa.vqagen.utils.qa_utils.get("length"))

    # Links
    opendrive_road_link = road.find("link")
    if opendrive_road_link is not None:
        parse_opendrive_road_link(newRoad, opendrive_road_link)

    # Type
    for opendrive_xml_road_type in road.findall("type"):
        parse_opendrive_road_type(newRoad, opendrive_xml_road_type)

    # Plan view
    for road_geometry in road.find("planView").findall("geometry"):
        parse_opendrive_road_geometry(newRoad, road_geometry)

    # Elevation profile
    road_elevation_profile = road.find("elevationProfile")
    if road_elevation_profile is not None:
        parse_opendrive_road_elevation_profile(newRoad, road_elevation_profile)

    # Lateral profile
    road_lateral_profile = road.find("lateralProfile")
    if road_lateral_profile is not None:
        parse_opendrive_road_lateral_profile(newRoad, road_lateral_profile)

    # Lanes
    lanes = road.find("lanes")

    if lanes is None:
        raise Exception("Road must have lanes element")

    # Lane offset
    for lane_offset in lanes.findall("laneOffset"):
        parse_opendrive_road_lane_offset(newRoad, lane_offset)

    # Lane sections
    for lane_section_id, lane_section in enumerate(road.find("lanes").findall("laneSection")):
        parse_opendrive_road_lane_section(newRoad, lane_section_id, lane_section)

    # Objects
    # TODO implementation

    # Signals
    # TODO implementation
    calculate_lane_section_lengths(newRoad)

    opendrive.roads.append(newRoad)


def calculate_lane_section_lengths(newRoad):
    """

    Args:
      newRoad:

    """
    # OpenDRIVE does not provide lane section lengths by itself, calculate them by ourselves
    for lane_section in newRoad.lanes.lane_sections:

        # Last lane section in road
        if lane_section.idx + 1 >= len(newRoad.lanes.lane_sections):
            lane_section.length = newRoad.planView.length - lane_section.sPos

        # All but the last lane section end at the succeeding one
        else:
            lane_section.length = (newRoad.lanes.lane_sections[lane_section.idx + 1].sPos - lane_section.sPos)

    # OpenDRIVE does not provide lane width lengths by itself, calculate them by ourselves
    for lane_section in newRoad.lanes.lane_sections:
        for lane in lane_section.allLanes:
            widthsPoses = np.array([x.start_offset for x in lane.widths] + [lane_section.length])
            widthsLengths = widthsPoses[1:] - widthsPoses[:-1]

            for widthIdx, width in enumerate(lane.widths):
                width.length = widthsLengths[widthIdx]


def parse_opendrive_header(opendrive, header):
    """

    Args:
      opendrive:
      header:

    """

    parsed_header = Header(
        vqa.vqagen.utils.qa_utils.get("revMajor"),
        vqa.vqagen.utils.qa_utils.get("revMinor"),
        vqa.vqagen.utils.qa_utils.get("name"),
        vqa.vqagen.utils.qa_utils.get("version"),
        vqa.vqagen.utils.qa_utils.get("date"),
        vqa.vqagen.utils.qa_utils.get("north"),
        vqa.vqagen.utils.qa_utils.get("south"),
        vqa.vqagen.utils.qa_utils.get("west"),
        vqa.vqagen.utils.qa_utils.get("vendor"),
    )
    # Reference
    if header.find("geoReference") is not None:
        pass
        # TODO not implemented

    opendrive.header = parsed_header


def parse_opendrive_junction(opendrive, junction):
    """

    Args:
      opendrive:
      junction:

    """
    newJunction = Junction()

    newJunction.id = int(vqa.vqagen.utils.qa_utils.get("id"))
    newJunction.name = str(vqa.vqagen.utils.qa_utils.get("name"))

    for connection in junction.findall("connection"):

        newConnection = JunctionConnection()

        newConnection.id = vqa.vqagen.utils.qa_utils.get("id")
        newConnection.incomingRoad = vqa.vqagen.utils.qa_utils.get("incomingRoad")
        newConnection.connectingRoad = vqa.vqagen.utils.qa_utils.get("connectingRoad")
        newConnection.contactPoint = vqa.vqagen.utils.qa_utils.get("contactPoint")

        for laneLink in connection.findall("laneLink"):
            newLaneLink = JunctionConnectionLaneLink()

            newLaneLink.fromId = vqa.vqagen.utils.qa_utils.get("from")
            newLaneLink.toId = vqa.vqagen.utils.qa_utils.get("to")

            newConnection.addLaneLink(newLaneLink)

        newJunction.addConnection(newConnection)

    opendrive.junctions.append(newJunction)
