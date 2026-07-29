"""
cordwell_corpus.py
Week 5, Lab 02: Cordwell Support-Doc Retrieval

Synthetic support knowledge base for Cordwell Home and Hardware, a fictional
retailer. Every product, part number, policy, and phone number here is invented.
Nothing in this file describes any real company, product, or customer.

The corpus is deliberately built in three length bands, because the whole point
of the lab is that one chunk size cannot serve all three:

  faq             about 60 to 160 words   short, self-contained answers
  troubleshooting about 350 to 700 words  procedural, one problem per document
  manual          about 1,800 to 3,400 words  many unrelated sections in one file

Documents are stored as a list of (heading, body) sections. Real manuals are
mostly boilerplate, so some sections are generated from templates. That is not
a shortcut, it is what the source material actually looks like, and it is what
makes large-chunk dilution happen for a real reason rather than a staged one.

Nothing here is random at runtime. The corpus is fully determined by the
literal text below plus a fixed seed used only for the boilerplate filler.
"""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Boilerplate section templates
#
# These are the sections that make a real manual long. They are topically far
# away from the sections that answer support questions, which is exactly why
# merging them into one 1,024 token vector destroys retrieval precision.
# ---------------------------------------------------------------------------

_SAFETY_TEMPLATE = """
Read all instructions before beginning installation. Failure to follow these
instructions can result in property damage, serious injury, or death. This
product must be installed in accordance with all applicable national and local
codes. If you are unsure whether your planned installation complies with local
code, stop and contact a licensed contractor before proceeding.

Disconnect power at the breaker panel before removing any cover plate. Confirm
the circuit is dead with a non contact voltage tester. Do not rely on a wall
switch to isolate the circuit. Lock out and tag the breaker if others have
access to the panel.

Wear approved eye protection during drilling, cutting, and fastening. Keep the
work area clear of children and pets. Do not operate the unit with any cover
panel removed. Do not modify the housing, the mounting plate, or any factory
wiring harness. Modification voids the Cordwell limited warranty described in
the warranty section of this manual and may create a fire hazard.

Retain this manual for the life of the product. A replacement copy is available
from any Cordwell store or from the Cordwell support portal.
"""

_TORQUE_TEMPLATE = """
Fastener torque values are listed below. Use a calibrated torque driver. Over
torquing the mounting screws is the most common cause of cracked housings
reported to Cordwell support.

Mounting plate to stud, number eight by one and one half inch wood screw,
{t1} inch pounds. Mounting plate to drywall anchor, number six by one inch pan
head, {t2} inch pounds. Cover plate retention screw, M3 by 6 millimeter,
{t3} inch pounds. Conduit fitting lock nut, {t4} inch pounds. Ground lug
terminal screw, {t5} inch pounds.

Apply torque in two passes. Bring all fasteners to half the listed value in a
diagonal pattern, then bring them to the full value in the same pattern. Do not
use an impact driver on any fastener listed in this table.
"""

_PARTS_TEMPLATE = """
The following parts are included in the carton. Confirm the contents against
this list before beginning. Missing parts are replaced at no charge within
thirty days of purchase.

One {product} main unit, part number {pn}-100. One wall mounting plate, part
number {pn}-210. One trim ring, part number {pn}-215. Four number eight by one
and one half inch wood screws, part number CW-HW-0081. Four conical drywall
anchors, part number CW-HW-0044. One wire nut assortment pack, part number
CW-HW-0110. One quick start card, part number {pn}-901. One warranty
registration card, part number CW-DOC-0002.

Replacement parts are ordered through any Cordwell store or the Cordwell parts
desk. Have the part number and the unit serial number available when ordering.
The serial number is printed on the label inside the battery compartment.
"""

_WARRANTY_TEMPLATE = """
Cordwell Home and Hardware warrants this product against defects in materials
and workmanship for a period of {years} years from the date of purchase, when
installed and operated in accordance with this manual. This warranty extends to
the original purchaser and is not transferable.

This warranty does not cover damage from improper installation, unauthorized
modification, power surge, flood, fire, freezing, insect or rodent intrusion,
or use of non Cordwell replacement parts. Consumable items including batteries,
filters, and gaskets are excluded. Labor for removal and reinstallation is not
covered unless the installation was performed by Cordwell Pro Install.

To make a claim, retain your proof of purchase and contact Cordwell support.
Cordwell will, at its option, repair the unit, replace the unit, or refund the
purchase price. Cordwell is not liable for incidental or consequential damages.
Some jurisdictions do not allow the exclusion of incidental or consequential
damages, so the preceding exclusion may not apply to you.
"""

_COMPLIANCE_TEMPLATE = """
This device complies with Part 15 of the applicable radio frequency rules.
Operation is subject to the following two conditions. This device may not cause
harmful interference. This device must accept any interference received,
including interference that may cause undesired operation.

Changes or modifications not expressly approved by Cordwell Home and Hardware
could void the user authority to operate this equipment. This equipment has been
tested and found to comply with the limits for a Class B digital device. These
limits are designed to provide reasonable protection against harmful
interference in a residential installation.

This equipment generates, uses, and can radiate radio frequency energy and, if
not installed and used in accordance with the instructions, may cause harmful
interference to radio communications. There is no guarantee that interference
will not occur in a particular installation.

Model {model}. Rated input {volts} volts alternating current, {hz} hertz.
Operating temperature range negative twenty to fifty degrees Celsius. Ingress
protection rating {ip}. Manufactured for Cordwell Home and Hardware.
"""

_SPEC_TEMPLATE = """
Electrical specifications. Supply voltage {volts} volts alternating current
plus or minus ten percent. Frequency {hz} hertz. Standby power draw {standby}
watts. Peak power draw {peak} watts. Inrush current {inrush} amperes for less
than one hundred milliseconds.

Mechanical specifications. Housing dimensions {w} by {h} by {d} millimeters.
Shipping weight {weight} kilograms. Mounting hole pattern eighty three
millimeter horizontal centers. Housing material ultraviolet stabilized
polycarbonate blend. Gasket material closed cell ethylene propylene diene
monomer.

Environmental specifications. Operating temperature negative twenty to fifty
degrees Celsius. Storage temperature negative thirty to seventy degrees
Celsius. Operating humidity five to ninety five percent non condensing.
Maximum installation altitude two thousand meters above sea level.
"""

_MAINTENANCE_TEMPLATE = """
Perform the following maintenance on the schedule listed. Cordwell support
records show that units on a maintenance schedule generate roughly one third
the service calls of units that are never inspected.

Monthly. Wipe the exterior housing with a damp cloth. Do not use solvent based
cleaners, which craze the polycarbonate housing. Confirm the status indicator
shows steady green.

Every six months. Inspect the gasket for cracking or compression set. Confirm
all mounting fasteners are at the torque values listed in the fastener section.
Inspect the supply wiring for discoloration at the terminals, which indicates a
loose connection generating heat.

Annually. Replace the backup battery regardless of remaining charge. Export the
event log from the Cordwell Home app and review it for repeated fault entries.
Repeated faults that clear on their own are the earliest warning of a failing
supply connection.
"""


def _boilerplate(kind: str, rng: random.Random, product: str, pn: str, model: str) -> str:
    """Fill one boilerplate template with plausible, stable-per-document values."""
    if kind == "safety":
        return _SAFETY_TEMPLATE
    if kind == "torque":
        return _TORQUE_TEMPLATE.format(
            t1=rng.choice([14, 16, 18]),
            t2=rng.choice([6, 8, 10]),
            t3=rng.choice([2, 3, 4]),
            t4=rng.choice([20, 24, 28]),
            t5=rng.choice([12, 15, 18]),
        )
    if kind == "parts":
        return _PARTS_TEMPLATE.format(product=product, pn=pn)
    if kind == "warranty":
        return _WARRANTY_TEMPLATE.format(years=rng.choice([2, 3, 5]))
    if kind == "compliance":
        return _COMPLIANCE_TEMPLATE.format(
            model=model,
            volts=rng.choice([120, 240]),
            hz=60,
            ip=rng.choice(["IP44", "IP54", "IP65"]),
        )
    if kind == "spec":
        return _SPEC_TEMPLATE.format(
            volts=rng.choice([120, 240]),
            hz=60,
            standby=rng.choice([1.2, 1.8, 2.4]),
            peak=rng.choice([45, 90, 180]),
            inrush=rng.choice([8, 12, 20]),
            w=rng.choice([102, 118, 134]),
            h=rng.choice([84, 96, 110]),
            d=rng.choice([28, 34, 41]),
            weight=rng.choice([0.6, 1.1, 2.3]),
        )
    if kind == "maintenance":
        return _MAINTENANCE_TEMPLATE
    raise ValueError(f"unknown boilerplate kind: {kind}")


# ---------------------------------------------------------------------------
# The answer-bearing sections. These are hand written.
#
# Naming convention below: a section whose heading starts with "ANSWER " is one
# that some labeled query depends on. The prefix is stripped before the corpus
# is built. It exists so an instructor can see the retrieval design at a glance.
# ---------------------------------------------------------------------------

_MANUALS: list[dict] = [
    {
        "document_id": "man_thermostat_t40",
        "source": "install_manual_thermalink_t40.md",
        "doc_type": "manual",
        "product_line": "climate",
        "created_at": "2025-02-11T09:00:00Z",
        "updated_at": "2026-03-20T14:45:00Z",
        "is_active": True,
        "product": "Cordwell ThermaLink T40 smart thermostat",
        "pn": "CW-TH40",
        "model": "T40",
        "seed": 101,
        "layout": [
            ("boiler", "safety"),
            ("boiler", "parts"),
            (
                "Removing the existing thermostat",
                """
                Turn off power to the heating and cooling system at the breaker
                panel. Photograph the existing wiring before disconnecting
                anything. The photograph is the single most useful thing you can
                do at this step, and support cannot reconstruct your wiring
                without it.

                Label each conductor with the supplied wire tags as you remove
                it. Do not rely on wire color. Conductor color conventions vary
                by installer and by decade, and a green conductor is not
                reliably a ground in residential heating and cooling wiring.

                Pull the conductor bundle forward through the wall opening and
                secure it with the supplied clip so it cannot fall back into the
                wall cavity. Seal the opening around the bundle with the supplied
                putty pad. An unsealed opening lets conditioned air from the wall
                cavity blow across the temperature sensor, which produces
                readings several degrees off from true room temperature.
                """,
            ),
            ("boiler", "torque"),
            (
                "Mounting bracket alignment",
                """
                Hold the mounting plate against the wall and use a torpedo level
                across the top edge. The plate has a molded level bubble, but the
                bubble reads the plate, not the wall, and a plate mounted on a
                bowed wall will read level while sitting crooked.

                Mark the four fastener positions with a pencil. Drill pilot holes
                sized for the anchor type in use. Do not overdrive the anchors.
                An anchor pulled proud of the drywall face prevents the plate from
                sitting flat, and a plate that does not sit flat will not accept
                the main unit without excessive force on the retention clips.

                Route the conductor bundle through the center opening of the
                plate before fastening. Fasten in a diagonal pattern to the torque
                values listed in the fastener section.
                """,
            ),
            (
                "Wire terminal reference",
                """
                The T40 terminal block accepts conductors from eighteen to
                twenty two American wire gauge. Strip length is nine millimeters.
                A conductor stripped longer than nine millimeters leaves bare
                copper outside the terminal, which is a short circuit risk against
                the adjacent terminal.

                Terminal R accepts twenty four volt hot from the transformer.
                Terminal C accepts twenty four volt common and is required for
                continuous power. Terminal W accepts the heating call. Terminal Y
                accepts the cooling call. Terminal G accepts the fan call.
                Terminal O and terminal B accept reversing valve control on heat
                pump systems and only one of the two is used on any given system.

                The C terminal is not optional on the T40. Earlier Cordwell
                thermostats could scavenge power through the heating call
                circuit. The T40 cannot, because the wireless radio draws more
                current than a scavenging circuit can supply without chattering
                the heating relay.
                """,
            ),
            ("boiler", "spec"),
            (
                "ANSWER Restoring the wireless connection after a power interruption",
                """
                After a power interruption the T40 restarts into a provisioning
                state and does not automatically rejoin the home wireless network.
                This is intentional. The T40 stores the network name but does not
                store the network credential in a form the radio can replay after
                a cold restart, so the credential must be re-presented from the
                Cordwell Home app.

                You have five minutes from the moment power is restored. During
                that window the status ring pulses amber and the unit advertises a
                temporary setup network. Open the Cordwell Home app, choose the
                T40 from the device list, and confirm the network when prompted.
                The ring turns steady green when the unit has rejoined.

                If the five minute window closes before you re-pair, the ring goes
                steady amber and the temporary setup network is withdrawn. Recover
                by holding the mode button for eight seconds until the ring flashes
                white, which reopens the window for another five minutes. The
                thermostat continues to run its stored schedule the entire time.
                Losing the wireless connection does not lose heating or cooling.

                A brownout, meaning a voltage sag that does not fully interrupt
                power, produces the same behavior as a full outage. Homes on long
                rural service drops see this after summer storms and report it as
                the thermostat randomly dropping offline. It is not random. It
                tracks the utility.
                """,
            ),
            (
                "ANSWER Wireless network requirements and radio behavior",
                """
                The T40 radio operates on the two point four gigahertz band only.
                It does not join five gigahertz networks and it does not join six
                gigahertz networks. Most current home routers publish all bands
                under a single network name, which is convenient for phones and a
                frequent source of failed thermostat setup, because the phone
                running the app may be attached to the five gigahertz radio while
                the thermostat can only see the two point four gigahertz radio.

                If setup fails repeatedly with the phone standing next to the
                thermostat, temporarily disable band steering on the router or
                publish a separate two point four gigahertz network name for
                setup, then re-enable band steering afterward.

                The T40 supports open, WPA2 Personal, and WPA3 Personal networks.
                It does not support enterprise authentication and it cannot
                complete a captive portal login. Networks with client isolation
                enabled will complete setup and then fail every subsequent app
                connection, because the phone and the thermostat can each reach
                the internet but cannot reach each other.

                Signal strength below negative seventy five decibel milliwatts at
                the thermostat produces intermittent disconnects that look like
                random dropouts. Read the current value in the app under device
                diagnostics before assuming a hardware fault.
                """,
            ),
            (
                "Scheduling and setpoint behavior",
                """
                The T40 supports seven day scheduling with up to six setpoint
                changes per day. Schedules are stored on the thermostat, not in
                the cloud, and continue to run with the network unavailable.

                A manual setpoint change holds until the next scheduled change
                unless permanent hold is selected. Permanent hold suspends the
                schedule until it is cleared. The most common support call about
                a thermostat ignoring its schedule is a permanent hold left on
                months earlier.

                Adaptive recovery starts the system early so the space reaches the
                scheduled temperature at the scheduled time rather than starting to
                work at that time. With adaptive recovery on, the system will
                appear to start heating before the schedule says it should. This is
                correct behavior.
                """,
            ),
            ("boiler", "maintenance"),
            (
                "Sensor calibration and offset",
                """
                The internal temperature sensor is factory calibrated to plus or
                minus point five degrees Celsius. A perceived error larger than
                that is almost always installation related rather than a bad
                sensor. Check the wall opening seal first, then check for a
                heat source such as a lamp or a television on the other side of
                the same wall.

                A calibration offset of plus or minus three degrees is available
                in the app under device settings. Applying an offset to mask an
                unsealed wall opening will produce a thermostat that reads
                correctly in mild weather and badly wrong in extreme weather,
                because the wall cavity temperature swings with outdoor
                conditions and the offset does not.
                """,
            ),
            ("boiler", "compliance"),
            ("boiler", "warranty"),
        ],
    },
    {
        "document_id": "man_water_heater_tx9",
        "source": "install_manual_aquasurge_tx9.md",
        "doc_type": "manual",
        "product_line": "plumbing",
        "created_at": "2025-05-02T09:00:00Z",
        "updated_at": "2026-02-09T11:15:00Z",
        "is_active": True,
        "product": "Cordwell AquaSurge TX9 tankless water heater",
        "pn": "CW-TX9",
        "model": "TX9",
        "seed": 202,
        "layout": [
            ("boiler", "safety"),
            (
                "Sizing the unit to the household",
                """
                A tankless heater is sized by flow rate at a required temperature
                rise, not by household size. Add the flow of every fixture that
                will run at once, then subtract the incoming water temperature
                from the desired output temperature to get the required rise.

                A standard shower head flows about six and a half liters per
                minute. A kitchen faucet flows about five and a half. A tub filler
                flows about fifteen. Incoming water temperature in northern
                climates drops to about five degrees Celsius in February, which is
                the month that undersized installations get discovered.

                The TX9 delivers its rated flow at a rise of thirty eight degrees
                Celsius. Every additional degree of rise costs flow. An installer
                who sizes against summer incoming temperature will deliver a unit
                that works beautifully until the first hard freeze.
                """,
            ),
            ("boiler", "parts"),
            (
                "Gas supply and venting",
                """
                The TX9 requires a dedicated gas line sized for its full rated
                input. Sharing a line with a furnace or a range will produce
                ignition faults under simultaneous demand. Verify supply pressure
                under full fire, not at rest. Static pressure tells you nothing
                about what the unit will see when it calls.

                Venting must be the manufacturer specified concentric or twin pipe
                assembly. Do not vent into a shared masonry chimney. Do not reduce
                the vent diameter. Do not exceed the equivalent length listed in
                the venting table, counting each ninety degree elbow as the listed
                equivalent length rather than as a fitting.

                Terminate the vent at the clearances listed. Termination too close
                to a window or a fresh air intake will recirculate combustion
                products and produce nuisance flame sense faults that look like an
                ignition problem and are actually a venting problem.
                """,
            ),
            ("boiler", "torque"),
            (
                "ANSWER Fault code reference E-4471 and related ignition faults",
                """
                Fault code E-4471 indicates a flame sense failure after a
                successful ignition attempt. The unit lit, the flame rod did not
                confirm it, and the gas valve closed on the safety timer. This is
                different from E-4470, which is a failure to light at all.

                Order of investigation for E-4471. First, remove and clean the
                flame rod with fine abrasive cloth. Oxide film on the rod is the
                cause in most reported cases. Do not use a wire brush, which
                embeds steel particles that accelerate re-fouling.

                Second, verify the ground path. Flame sense works by rectifying a
                current through the flame to chassis ground. A high resistance
                ground produces a signal too weak to confirm even with a perfectly
                clean rod. Measure from the burner assembly to the equipment ground
                lug. Anything above one ohm should be corrected.

                Third, verify combustion air. A vent termination that recirculates
                exhaust starves the flame of oxygen and drops the sense current
                below threshold. This presents as an E-4471 that only occurs on
                calm days or with wind from one direction.

                Related codes. E-4472 indicates flame present with the gas valve
                commanded closed, which is a valve fault and requires immediate
                shutdown. E-4473 indicates repeated E-4471 events within one hour
                and locks the unit out until it is manually reset at the panel.
                """,
            ),
            ("boiler", "spec"),
            (
                "ANSWER Descaling and hard water service",
                """
                Scale accumulation on the heat exchanger is the dominant failure
                mode for tankless units in hard water regions. Scale is an
                insulator. As it builds, the burner runs longer to hit the same
                outlet temperature, which raises exchanger surface temperature,
                which deposits scale faster. The process accelerates.

                Descale annually at water hardness below one hundred fifty parts
                per million, every six months from one hundred fifty to two
                hundred fifty, and quarterly above two hundred fifty. If you do
                not know the hardness, a test strip from any Cordwell store costs
                less than one service call.

                The procedure. Close both isolation valves. Connect a circulation
                pump to the service ports. Circulate four liters of food grade
                descaling solution for forty five minutes. Drain, then flush with
                clean water for five minutes with both isolation valves open.
                Reopen the supply and purge air from the fixtures before restoring
                power.

                Symptoms of overdue descaling, in the order they appear. Outlet
                temperature drifts below setpoint under high flow. The unit
                short cycles at low flow. A rumbling or kettling sound develops
                during operation. Finally the unit throws an over temperature
                fault and locks out.
                """,
            ),
            (
                "Freeze protection",
                """
                The TX9 contains internal freeze protection that energizes below
                three degrees Celsius, and it requires electrical power to work. A
                winter power outage removes freeze protection at exactly the moment
                it is needed.

                Units installed in unconditioned space in freezing climates must be
                drained if power will be out for an extended period, or supplied
                from a backup circuit. Freeze damage to the heat exchanger is not
                covered by the warranty.

                Exposed supply piping within one meter of the unit should be
                insulated regardless of freeze protection, because the internal
                protection heats the unit and not the piping.
                """,
            ),
            ("boiler", "maintenance"),
            ("boiler", "compliance"),
            ("boiler", "warranty"),
        ],
    },
    {
        "document_id": "man_garage_g7",
        "source": "install_manual_gateway_g7.md",
        "doc_type": "manual",
        "product_line": "access",
        "created_at": "2025-03-18T09:00:00Z",
        "updated_at": "2025-11-30T16:20:00Z",
        "is_active": True,
        "product": "Cordwell GateWay G7 smart garage door opener",
        "pn": "CW-G7",
        "model": "G7",
        "seed": 303,
        "layout": [
            ("boiler", "safety"),
            ("boiler", "parts"),
            (
                "Rail assembly and headroom",
                """
                The G7 requires fifty millimeters of headroom above the highest
                point of door travel plus the rail depth. Measure at the highest
                point, not at the door top edge, because a sectional door rises
                above its closed top edge as it rounds the curve.

                Assemble the rail on the floor, not overhead. Join the rail
                sections with the supplied splice plates and confirm the joint is
                flush by running a finger along the inside surface. A proud joint
                will chatter the trolley and will eventually strip the belt.

                Support the rail at the header before releasing it. The motor head
                weighs enough to bend the rail if it is allowed to hang from one
                end during assembly.
                """,
            ),
            (
                "ANSWER Pairing a remote or keypad to the opener",
                """
                The G7 stores up to fifty paired accessories. Pairing is done at
                the motor head, not in the app, because the pairing exchange uses
                the short range accessory radio rather than the home network.

                Press and release the learn button on the motor head. The learn
                indicator lights steady blue for thirty seconds. Within that
                window, press and hold the button on the remote until the motor
                head indicator flashes twice, then release. Test the remote before
                the window closes.

                For a wireless keypad, enter your chosen four digit code and press
                enter while the learn indicator is lit. The motor head flashes
                twice to confirm.

                If the learn indicator does not light, the motor head has not
                exited its startup self test. Wait thirty seconds after power up
                before pairing.

                To clear all paired accessories, press and hold the learn button
                until the indicator goes dark, about six seconds. This erases
                every remote and keypad, including ones you still want, and there
                is no partial erase. Clearing and re-pairing is the correct
                response to a lost remote.
                """,
            ),
            ("boiler", "torque"),
            (
                "ANSWER Safety reversing sensor alignment and obstruction faults",
                """
                The photo eye sensors mount no higher than one hundred fifty
                millimeters above the floor on both tracks. Mounting them higher
                creates a gap under the beam large enough for a child to crawl
                through, and it is a code violation in every jurisdiction Cordwell
                sells into.

                The sending unit indicator is amber and is lit whenever it has
                power. The receiving unit indicator is green and is lit only when
                it sees the beam. A dark green indicator means the beam is broken,
                misaligned, or the receiving unit has lost power.

                Alignment procedure. Loosen the receiving unit bracket. Sweep it
                slowly through its arc until the green indicator lights, then
                continue sweeping to find the far edge of the arc where it goes
                dark again. Set the bracket at the midpoint of the lit arc and
                tighten. Setting it at the first point where the light comes on
                produces an alignment that fails as soon as the track flexes.

                A door that closes partway and reverses, with both indicators lit,
                is not a sensor fault. That is the force setting. Direct sunlight
                falling on the receiving lens produces intermittent phantom
                obstructions in late afternoon only, which is diagnosed by shading
                the lens with a hand and retrying.
                """,
            ),
            ("boiler", "spec"),
            (
                "Force and travel limit setting",
                """
                Travel limits are set with the up and down buttons on the motor
                head with the unit in limit setting mode. Run the door to fully
                closed and press set. Run it to fully open and press set. The unit
                exits limit setting mode automatically.

                Force is learned on the next two full cycles after limits are set.
                Do not obstruct the door during these cycles and do not hold the
                wall control. A force value learned against a partially seized
                spring will be too high, and the door will not reverse on
                obstruction as it must.

                Verify the reversal test after any limit or force change. Lay a
                fifty millimeter block flat on the floor in the door path and run
                the door closed. The door must contact the block and reverse fully
                open. A door that stops without reversing, or that continues
                pressing, must be taken out of service until it passes.
                """,
            ),
            ("boiler", "maintenance"),
            ("boiler", "compliance"),
            ("boiler", "warranty"),
        ],
    },
    {
        "document_id": "man_mower_rm3",
        "source": "install_manual_turfmate_rm3.md",
        "doc_type": "manual",
        "product_line": "outdoor",
        "created_at": "2025-04-22T09:00:00Z",
        "updated_at": "2026-04-14T10:05:00Z",
        "is_active": True,
        "product": "Cordwell TurfMate RM3 robotic mower",
        "pn": "CW-RM3",
        "model": "RM3",
        "seed": 404,
        "layout": [
            ("boiler", "safety"),
            ("boiler", "parts"),
            (
                "ANSWER Boundary wire layout and signal loss",
                """
                The boundary wire defines the mowing area by carrying a low
                frequency signal the mower detects. The mower stays inside the
                loop. There is no satellite positioning involved and no map stored
                in the mower.

                Lay the wire two hundred millimeters inside any hard edge such as
                a driveway or a patio, and three hundred millimeters inside any
                soft edge such as a flower bed the mower should not enter. Around
                an obstacle island, run the wire out and back with the two runs
                touching, which cancels the signal along the corridor and lets the
                mower cross it.

                Signal loss is reported as boundary wire not found. Causes, in
                order of how often Cordwell support sees them. A break in the wire,
                usually at a splice or where an aerator or edger crossed it. A
                splice made with electrical tape rather than a gel filled
                connector, which corrodes through in one season. The loop exceeding
                the maximum length of two hundred fifty meters. The two loop ends
                reversed at the base station, which produces a signal the mower
                reads as inverted and refuses to trust.

                Find a break by halving. Disconnect one loop end and probe with the
                supplied wire locator at the midpoint of the run. The break is in
                whichever half still fails. Repeat until the segment is short
                enough to inspect visually.
                """,
            ),
            (
                "Base station placement",
                """
                The base station requires one meter of clear approach in front and
                point five meters clear on each side. The mower approaches the
                station along the boundary wire and needs the run in front of the
                station to be straight for at least three meters.

                Site the station on level ground, in shade if possible, and not in
                a location that collects standing water. Battery charge acceptance
                falls sharply above forty degrees Celsius, and a station in full
                afternoon sun in summer will produce a mower that docks and then
                charges slowly enough to miss its next scheduled cut.

                The station must be powered from a ground fault protected outdoor
                circuit. Do not extend the low voltage cable beyond the supplied
                length.
                """,
            ),
            ("boiler", "torque"),
            (
                "ANSWER Blade replacement and cut quality",
                """
                The RM3 uses three pivoting razor blades on a rotating disc rather
                than a single fixed blade. The pivot is deliberate. A blade that
                strikes a buried stone swings back instead of transmitting the
                shock to the spindle bearing.

                Replace all three blades and all three screws together, every two
                months during the growing season or immediately after any strike
                that leaves a visible nick. Replacing one blade of three leaves the
                disc out of balance, and the resulting vibration destroys the
                spindle bearing within one season.

                Torque the blade screws to two and a half newton meters. Do not
                reuse blade screws. The supplied screws have a pre applied thread
                locking compound that is single use.

                Poor cut quality with sharp blades is a cutting height problem, not
                a blade problem. The RM3 removes a small amount frequently and
                cannot recover an overgrown lawn. Cut with a conventional mower
                first, then set the RM3 no more than ten millimeters below the
                current grass height and step it down over two weeks.
                """,
            ),
            ("boiler", "spec"),
            (
                "Scheduling and rain handling",
                """
                Schedule the RM3 for the total area rather than for a time of day
                that feels right. The RM3 covers roughly one hundred square meters
                per hour of active cutting including its return trips to charge.

                The rain sensor pauses the schedule and sends the mower to dock.
                Cutting wet grass clogs the discharge and leaves clumps that smother
                the lawn beneath them. The paused time is not automatically made up.
                In a wet week, extend the schedule manually rather than assuming the
                mower caught up.

                Do not schedule overnight operation in areas with hedgehog or
                tortoise activity. Cordwell recommends a daylight only schedule
                where local wildlife is a consideration.
                """,
            ),
            ("boiler", "maintenance"),
            ("boiler", "compliance"),
            ("boiler", "warranty"),
        ],
    },
    {
        "document_id": "man_deadbolt_l2",
        "source": "install_manual_boltguard_l2.md",
        "doc_type": "manual",
        "product_line": "access",
        "created_at": "2025-06-30T09:00:00Z",
        "updated_at": "2026-01-27T13:40:00Z",
        "is_active": True,
        "product": "Cordwell BoltGuard L2 smart deadbolt",
        "pn": "CW-L2",
        "model": "L2",
        "seed": 505,
        "layout": [
            ("boiler", "safety"),
            ("boiler", "parts"),
            (
                "Door preparation and backset",
                """
                The L2 fits standard door preparation with a fifty four millimeter
                bore and a twenty five millimeter edge bore. Backset is adjustable
                between sixty and seventy millimeters by rotating the latch collar.
                Confirm your backset before drilling anything.

                Door thickness range is thirty five to forty five millimeters
                standard. Thicker doors require the extension kit, part number
                CW-L2-330, which includes a longer tailpiece and longer through
                bolts.

                The strike plate must be shimmed or mortised so the bolt enters
                without contact. A bolt that rubs the strike will operate by hand
                and will fail under motor drive, because the motor has far less
                torque available than a thumb turn.
                """,
            ),
            (
                "ANSWER Battery life, low battery behavior, and emergency power",
                """
                The L2 runs on four AA alkaline cells. Expected life is roughly
                nine months at ten cycles per day with the wireless bridge
                disabled, and roughly four months with the bridge enabled and
                polling every thirty seconds.

                Do not use lithium primary cells. They hold a high voltage almost
                to exhaustion and then collapse, which defeats the low battery
                warning entirely. Do not mix chemistries or mix old and new cells.
                Rechargeable nickel metal hydride cells work but their lower cell
                voltage triggers the low battery warning at roughly half of usable
                capacity.

                Low battery warning sequence. At approximately twenty percent
                remaining the keypad flashes amber after each successful unlock and
                the app posts a notification. At approximately five percent the
                unit disables the wireless bridge to conserve what remains for the
                motor. At depletion the motor will not drive and the keypad is
                dark.

                A fully depleted L2 is opened from outside with the mechanical key
                override under the keypad cover, or by applying a nine volt battery
                to the emergency contacts on the underside of the exterior escutcheon
                while entering a valid code. Keep the mechanical key somewhere other
                than inside the locked building.
                """,
            ),
            ("boiler", "torque"),
            (
                "ANSWER Access codes, guest codes, and audit history",
                """
                The L2 stores up to two hundred fifty access codes. Codes are four
                to eight digits. Codes are stored on the lock, so a code entered at
                the keypad works with the network down.

                Guest codes carry a schedule. A recurring guest code is active on
                selected days between selected hours. A single use guest code is
                consumed on first successful entry. A date bounded guest code is
                active between two dates and then expires. Schedules are evaluated
                against the clock on the lock, which drifts roughly one minute per
                month and is corrected whenever the bridge is connected.

                A lock that has been offline for a long period will evaluate guest
                schedules against a drifted clock. If a guest code is rejected at
                the edge of its window on a lock that has been offline, check the
                clock before assuming a code problem.

                The audit history holds the last five hundred events on the lock
                and is retained indefinitely in the app once synced. Events record
                which code was used, not who used it, so shared codes destroy the
                usefulness of the audit history. Issue one code per person.
                """,
            ),
            ("boiler", "spec"),
            ("boiler", "maintenance"),
            ("boiler", "compliance"),
            ("boiler", "warranty"),
        ],
    },
    {
        "document_id": "man_sump_stormshield",
        "source": "install_manual_stormshield_sump.md",
        "doc_type": "manual",
        "product_line": "plumbing",
        "created_at": "2025-01-20T09:00:00Z",
        "updated_at": "2025-09-08T08:30:00Z",
        "is_active": True,
        "product": "Cordwell StormShield sump pump",
        "pn": "CW-SS",
        "model": "SS-Series",
        "seed": 606,
        "layout": [
            ("boiler", "safety"),
            ("boiler", "parts"),
            (
                "Pit sizing and pump placement",
                """
                The pit must be at least four hundred fifty millimeters in diameter
                and six hundred millimeters deep. A pit smaller than this forces the
                pump to cycle too frequently, and cycle count rather than run hours
                is what wears out a sump pump.

                Set the pump on a solid base, not on loose gravel. A pump that
                settles will draw silt, and silt in the impeller is the second most
                common cause of premature failure after short cycling.

                Leave clearance around the float so it can swing through its full
                arc without contacting the pit wall, the discharge pipe, or the
                power cord. A restrained float is the single most common cause of a
                pump that either never starts or never stops.
                """,
            ),
            ("boiler", "torque"),
            (
                "ANSWER Short cycling diagnosis",
                """
                Short cycling means the pump starts and stops repeatedly over a
                short interval, often several times per minute. It is the fastest
                way to destroy a sump pump and it always has a mechanical cause.

                Cause one, a missing or failed check valve. Without a working check
                valve the water in the vertical discharge run falls back into the
                pit the moment the pump stops, which refills the pit and starts the
                pump again. Diagnose by listening for a distinct gurgle immediately
                after shutoff.

                Cause two, a float switch set with too narrow a differential
                between the on and off levels. Widen the differential to at least
                two hundred millimeters where the pit depth allows it.

                Cause three, a pit that is too small for the inflow rate. This is
                an installation error and is corrected by enlarging the pit or by
                adding a second pit, not by adjusting the pump.

                Cause four, a check valve installed backwards. This presents as a
                pump that runs continuously and moves no water rather than as
                classic short cycling, and it is worth ruling out early because it
                is free to check.
                """,
            ),
            ("boiler", "spec"),
            (
                "Discharge and freeze considerations",
                """
                Discharge must terminate at least three meters from the foundation
                and must not discharge into a sanitary sewer, which is prohibited in
                most jurisdictions and can flood a neighbor during a regional storm.

                A freeze in the discharge line makes the pump run against a closed
                pipe. Fit a freeze relief opening in the discharge inside the
                building so that a frozen exterior run still has somewhere to go.

                Install a union in the discharge line above the check valve. Without
                a union, servicing the pump requires cutting the pipe.
                """,
            ),
            ("boiler", "maintenance"),
            ("boiler", "compliance"),
            ("boiler", "warranty"),
        ],
    },
]


_TROUBLESHOOTING: list[dict] = [
    {
        "document_id": "ts_thermostat_wifi",
        "source": "troubleshoot_thermostat_wifi.md",
        "doc_type": "troubleshooting",
        "product_line": "climate",
        "created_at": "2025-08-14T10:00:00Z",
        "updated_at": "2026-05-19T09:12:00Z",
        "is_active": True,
        "text": """
        Thermostat drops off the network after a power outage

        Symptom. The Cordwell ThermaLink T40 is unreachable from the app after a
        storm, a breaker trip, or a utility outage. Heating and cooling still
        work. The status ring is amber rather than green.

        Cause. The T40 restarts into a provisioning state after any loss of
        supply power and waits for the network credential to be re-presented from
        the app. It does not silently rejoin on its own. The five minute
        re-pairing window opens the moment power is restored and closes whether or
        not anyone was home to use it.

        Resolution. Hold the mode button for eight seconds until the status ring
        flashes white. That reopens the five minute window. Open the Cordwell Home
        app, select the T40, and confirm the network when prompted. The ring turns
        steady green on success.

        If it fails. Confirm the phone is on the two point four gigahertz network
        rather than the five gigahertz network. The T40 radio is two point four
        gigahertz only. On a router that publishes all bands under one name, the
        phone will usually attach to five gigahertz and the pairing exchange will
        never complete. Temporarily disabling band steering resolves this.

        Confirm client isolation is off on the router. With client isolation on,
        setup completes and every later app connection fails, which reads as an
        intermittent fault and is not.

        Check the signal strength reading in the app under device diagnostics.
        Below negative seventy five decibel milliwatts, expect repeated dropouts
        that have nothing to do with the outage.

        Repeat offenders. A home that reports this after every summer storm is
        usually seeing brownouts rather than full outages. A voltage sag restarts
        the thermostat exactly as a full outage does. Recommend a small
        uninterruptible supply on the thermostat common circuit rather than
        replacing the unit. Cordwell support has never confirmed a hardware fault
        from this symptom pattern.
        """,
    },
    {
        "document_id": "ts_water_heater_codes",
        "source": "troubleshoot_water_heater_codes.md",
        "doc_type": "troubleshooting",
        "product_line": "plumbing",
        "created_at": "2025-07-05T10:00:00Z",
        "updated_at": "2026-05-02T15:45:00Z",
        "is_active": True,
        "text": """
        AquaSurge TX9 fault code quick reference

        This is the field reference. The installation manual carries the full
        diagnostic sequence for each code.

        E-4470. Failure to light. No flame detected during the ignition attempt.
        Check gas supply pressure under full fire, check the igniter gap, check
        for a closed manual shutoff someone forgot after service.

        E-4471. Flame sense failure after successful ignition. The unit lit and
        the flame rod did not confirm. Clean the flame rod with fine abrasive
        cloth, verify the ground path measures under one ohm from burner assembly
        to equipment ground, then verify the vent termination is not recirculating
        exhaust. Most reported cases are an oxide film on the rod.

        E-4472. Flame present with the gas valve commanded closed. Shut the unit
        down at the manual valve and do not return it to service. This is a gas
        valve fault.

        E-4473. Three or more E-4471 events within one hour. The unit has locked
        out and requires a manual reset at the panel after the underlying E-4471
        cause is corrected.

        E-5510. Outlet temperature above the over temperature limit. Almost always
        scale on the heat exchanger. Descale before assuming a sensor fault.

        E-5511. Outlet temperature sensor open circuit. Check the connector at the
        control board before replacing the sensor.

        E-6120. Flow sensor reports no flow with a demand present. Check for a
        closed isolation valve, then check the flow sensor impeller for scale.

        E-6121. Flow below the minimum activation threshold. The unit will not
        fire below about one point eight liters per minute. A single low flow
        aerator can put a fixture below the threshold, and the customer reports it
        as no hot water at that sink only.

        Escalate to Cordwell technical support for any code not listed here, and
        for any E-4472 regardless of whether it cleared on its own.
        """,
    },
    {
        "document_id": "ts_garage_remote",
        "source": "troubleshoot_garage_remote.md",
        "doc_type": "troubleshooting",
        "product_line": "access",
        "created_at": "2025-09-01T10:00:00Z",
        "updated_at": "2026-03-11T12:00:00Z",
        "is_active": True,
        "text": """
        GateWay G7 remote or keypad stops working

        Symptom. A remote or wireless keypad that used to work no longer operates
        the door. The wall control still works.

        First check the obvious. Replace the remote battery. A weak coin cell
        produces a remote that works at two meters and fails at ten, which reads
        as intermittent and is not.

        Confirm the motor head is receiving. Watch the learn indicator while
        pressing the remote. A brief flicker means the signal arrived and was
        rejected as unpaired. No flicker at all means the signal is not arriving.

        Re-pair the accessory. Press and release the learn button on the motor
        head, wait for the steady blue indicator, then hold the remote button
        until the head flashes twice. If the head has reached its fifty accessory
        limit, the pairing will silently fail. Clear all accessories with a six
        second hold on the learn button and re-pair everything you still use.

        Interference. The accessory radio shares its band with some LED fixtures
        and some low quality power supplies. If range collapsed suddenly and
        nothing else changed, ask what was recently installed in the garage. LED
        shop lights are the most common culprit Cordwell support sees, and the
        test is to switch them off and retry from the driveway.

        A door that will not close and reverses immediately is not a remote
        problem. That is the safety reversing sensors. Check the green indicator
        on the receiving sensor.

        A door that closes partway and reverses with both sensor indicators lit is
        a force setting problem, not a sensor problem, and it is a signal that
        something mechanical has changed in the door.
        """,
    },
    {
        "document_id": "ts_mower_boundary",
        "source": "troubleshoot_mower_boundary.md",
        "doc_type": "troubleshooting",
        "product_line": "outdoor",
        "created_at": "2025-06-11T10:00:00Z",
        "updated_at": "2026-04-30T14:22:00Z",
        "is_active": True,
        "text": """
        TurfMate RM3 reports boundary wire not found

        Symptom. The mower will not leave the base station and the app shows
        boundary wire not found. The base station indicator is red or flashing.

        This is a loop continuity problem in almost every reported case. The mower
        is fine.

        Check the base station terminals first. The two loop ends must be seated
        fully and must not be reversed. A reversed loop produces an inverted
        signal that the mower detects and refuses to trust, which reports as the
        same fault as a clean break.

        Check every splice. A splice made with electrical tape rather than a gel
        filled connector will corrode through within one season, and it will fail
        after rain rather than continuously, which makes it look intermittent.

        Check where the loop crosses anything that gets serviced. Aerator tines,
        edger blades, and fence post augers cut boundary wire regularly. Walk the
        loop looking for recent ground disturbance before you start probing.

        Find the break by halving. Disconnect one end of the loop at the station.
        Use the wire locator at the midpoint of the run. Whichever half still
        fails contains the break. Repeat on that half. Four or five halvings will
        narrow two hundred meters down to a few meters of visual inspection.

        Confirm loop length. The maximum is two hundred fifty meters. A yard that
        worked for two seasons and then failed after the owner extended the loop
        around a new bed is over length, not broken.

        Repair with gel filled connectors only. Splice, then bury the splice
        rather than leaving it at the surface where a mower wheel will find it.
        """,
    },
    {
        "document_id": "ts_deadbolt_battery",
        "source": "troubleshoot_deadbolt_battery.md",
        "doc_type": "troubleshooting",
        "product_line": "access",
        "created_at": "2025-10-02T10:00:00Z",
        "updated_at": "2026-02-18T09:55:00Z",
        "is_active": True,
        "text": """
        BoltGuard L2 batteries drain faster than expected

        Symptom. The customer reports replacing batteries every six to eight weeks
        against an expected life of several months.

        Ask what cells are in it. Lithium primary cells are the most common cause
        of a confusing battery report. They hold high voltage until they collapse,
        so the low battery warning never fires and the customer experiences the
        lock as fine and then suddenly dead. Alkaline cells are specified.

        Ask about the wireless bridge polling interval. A bridge polling every
        thirty seconds cuts expected battery life roughly in half compared with the
        bridge disabled. Moving to a five minute polling interval recovers most of
        that without any practical loss of responsiveness.

        Check the bolt for mechanical drag. This is the cause that actually matters
        and it is the one most often missed. A bolt that rubs the strike draws
        motor current for the entire throw instead of only at the ends. Operate the
        lock with the door open and then with the door closed. If it is noticeably
        slower or louder with the door closed, the strike needs shimming or
        mortising. Fixing the strike frequently doubles battery life.

        Check for repeated failed unlock attempts in the audit history. A guest
        code that has expired but is still being tried several times a day drives
        motor and radio activity that shows up as battery drain with no obvious
        cause.

        Cold matters. Alkaline capacity falls sharply below freezing. An exterior
        door on an unheated porch in a cold climate will show shorter life every
        winter, and that is chemistry rather than a fault.
        """,
    },
    {
        "document_id": "ts_sump_cycling",
        "source": "troubleshoot_sump_cycling.md",
        "doc_type": "troubleshooting",
        "product_line": "plumbing",
        "created_at": "2025-05-27T10:00:00Z",
        "updated_at": "2025-12-15T11:30:00Z",
        "is_active": True,
        "text": """
        StormShield pump runs constantly or cycles rapidly

        Symptom. The pump starts and stops every few seconds, or runs without
        stopping. Either pattern will destroy the pump quickly.

        Rapid cycling. Check the check valve first. Without a working check valve
        the water standing in the vertical discharge falls back into the pit as
        soon as the pump stops, refilling the pit and restarting the pump. The
        diagnostic sound is a distinct gurgle right after shutoff.

        Then check the float differential. The gap between the on level and the off
        level should be at least two hundred millimeters. A narrow differential
        cycles the pump many more times for the same volume of water, and cycle
        count is what wears the pump out.

        Then check pit size. A pit under four hundred fifty millimeters in diameter
        cannot hold enough water between cycles at any meaningful inflow rate. This
        is an installation defect and no pump setting will fix it.

        Continuous running with no water moving. Check whether the check valve was
        installed backwards. A reversed check valve gives you a pump that runs, is
        warm, and moves nothing.

        Continuous running with water moving. The inflow genuinely exceeds what one
        pump can handle, or the float is physically restrained in the on position by
        the discharge pipe, the cord, or the pit wall.

        Never solve a cycling problem by restricting the discharge. It raises head
        pressure, overheats the motor, and voids the warranty.
        """,
    },
    {
        "document_id": "ts_lighting_dim",
        "source": "troubleshoot_landscape_lighting.md",
        "doc_type": "troubleshooting",
        "product_line": "outdoor",
        "created_at": "2025-04-08T10:00:00Z",
        "updated_at": "2025-10-21T16:10:00Z",
        "is_active": True,
        "text": """
        LumaPath landscape fixtures dim toward the end of the run

        Symptom. Fixtures near the transformer are at full brightness and fixtures
        at the far end of the run are visibly dimmer.

        This is voltage drop and it is a wiring layout problem, not a fixture
        problem. Low voltage landscape wiring loses voltage along its length in
        proportion to the current it carries and the resistance of the conductor.
        By the time you reach the last fixture on a long daisy chain, there may not
        be enough left.

        Measure at the fixture rather than at the transformer. A twelve volt system
        should read between ten point eight and twelve volts at every fixture.
        Below ten point eight, light emitting diode drivers begin to reduce output
        and eventually flicker.

        Fix one. Change the topology. A daisy chain is the worst case. A hub and
        spoke layout, where each fixture or small group runs back to a common
        point, distributes the drop evenly. A loop layout, fed from both ends,
        halves the effective run length.

        Fix two. Increase the conductor size. Going from sixteen gauge to twelve
        gauge cuts the resistance of the run substantially and is usually cheaper
        than adding a second transformer.

        Fix three. Use the higher voltage taps on the transformer if it has them.
        A thirteen or fourteen volt tap compensates for a known drop. Do not use a
        high tap to mask a wiring fault, because the near fixtures will then be
        overdriven and will fail early.

        Do not add fixtures to an existing run without recalculating. The most
        common cause of a system that worked and then dimmed is three fixtures
        added to the end of a run that was already at its limit.
        """,
    },
    {
        "document_id": "ts_app_pairing",
        "source": "troubleshoot_app_device_pairing.md",
        "doc_type": "troubleshooting",
        "product_line": "platform",
        "created_at": "2025-11-12T10:00:00Z",
        "updated_at": "2026-05-25T08:40:00Z",
        "is_active": True,
        "text": """
        Cordwell Home app cannot find a device during setup

        This applies to every Cordwell connected product. Work the list in order.

        Confirm the phone is on the two point four gigahertz network. Every
        Cordwell device radio is two point four gigahertz only. Routers that
        publish all bands under one network name will usually put a modern phone
        on five gigahertz, and the setup exchange requires both ends on the same
        radio. Temporarily disable band steering, or publish a separate two point
        four gigahertz name for setup.

        Confirm client isolation is disabled. With it enabled, setup can complete
        and every subsequent connection fails. This produces a support call that
        sounds like an intermittent hardware fault and is a router setting.

        Confirm the device is actually in setup mode. Most Cordwell devices open a
        time bounded setup window rather than advertising continuously. The window
        is five minutes on the ThermaLink T40 and ten minutes on most other
        products. Reopen it with the documented button hold for that product.

        Confirm the phone has local network permission granted to the Cordwell Home
        app. Device discovery uses local network broadcast, and a denied permission
        produces a device list that stays empty with no error message.

        Confirm no virtual private network is active on the phone. A VPN routes the
        discovery broadcast away from the local network and the device will never
        appear.

        Guest networks and mesh extenders. Devices set up on a guest network are
        frequently isolated from the main network by design. Set up on the main
        network. Mesh systems that place the phone and the device on different
        nodes usually still work, but if setup fails repeatedly, stand next to the
        main router with both.
        """,
    },
]


_FAQS: list[dict] = [
    {
        "document_id": "faq_returns",
        "source": "faq_returns.md",
        "doc_type": "faq",
        "product_line": "policy",
        "created_at": "2025-01-05T08:00:00Z",
        "updated_at": "2026-01-08T08:00:00Z",
        "is_active": True,
        "text": """
        What is the Cordwell return policy?

        Most items can be returned within ninety days of purchase with proof of
        purchase. Cordwell Pro account holders have three hundred sixty five days.
        Refunds go back to the original payment method within five to seven
        business days after the item is received.

        Special order items, cut to length materials, and clearance items marked
        final sale cannot be returned. Major appliances have a forty eight hour
        return window from delivery. Items showing installation wear are handled as
        warranty claims rather than returns.
        """,
    },
    {
        "document_id": "faq_delivery",
        "source": "faq_delivery.md",
        "doc_type": "faq",
        "product_line": "policy",
        "created_at": "2025-01-05T08:00:00Z",
        "updated_at": "2025-11-02T08:00:00Z",
        "is_active": True,
        "text": """
        How does Cordwell delivery scheduling work?

        Standard delivery is scheduled in four hour windows and you receive a text
        with a narrowed two hour window the morning of delivery. Same day delivery
        is available in most metropolitan areas for orders placed before eleven in
        the morning.

        Large item delivery includes placement in the room of your choice. It does
        not include unpacking, assembly, connection to plumbing or electrical, or
        removal of an old appliance unless haul away was purchased.
        """,
    },
    {
        "document_id": "faq_warranty_registration",
        "source": "faq_warranty_registration.md",
        "doc_type": "faq",
        "product_line": "policy",
        "created_at": "2025-02-14T08:00:00Z",
        "updated_at": "2025-08-19T08:00:00Z",
        "is_active": True,
        "text": """
        Do I need to register my product for the warranty to apply?

        No. The Cordwell limited warranty applies from the date of purchase whether
        or not you register. Registration is optional and exists so we can reach you
        about a safety notice and so a claim can be processed without a receipt.

        Register in the Cordwell Home app or on the support portal using the model
        number and serial number from the product label. Registration does not
        extend the warranty term.
        """,
    },
    {
        "document_id": "faq_pro_account",
        "source": "faq_pro_account.md",
        "doc_type": "faq",
        "product_line": "policy",
        "created_at": "2025-03-03T08:00:00Z",
        "updated_at": "2026-02-25T08:00:00Z",
        "is_active": True,
        "text": """
        What does a Cordwell Pro account include?

        Volume pricing on qualifying orders, an extended three hundred sixty five
        day return window, dedicated pro desk checkout, job site delivery, and
        consolidated monthly invoicing with net thirty terms on approved credit.

        Pro accounts also get purchase history export for job costing and the
        ability to attach a job name to every order at checkout. There is no annual
        fee. A business license or tax identification number is required to open an
        account.
        """,
    },
    {
        "document_id": "faq_store_pickup",
        "source": "faq_store_pickup.md",
        "doc_type": "faq",
        "product_line": "policy",
        "created_at": "2025-01-19T08:00:00Z",
        "updated_at": "2025-09-30T08:00:00Z",
        "is_active": True,
        "text": """
        How does store pickup work?

        Order online and choose pickup at your preferred store. You receive a text
        when the order is staged, usually within two hours during store hours. Bring
        the order number and a photo identification to the pickup desk near the
        front entrance.

        Orders are held for seven days. After seven days an unclaimed order is
        returned to stock and refunded automatically. Someone else can pick up on
        your behalf if you add them as an authorized pickup contact on the order.
        """,
    },
    {
        "document_id": "faq_price_match",
        "source": "faq_price_match.md",
        "doc_type": "faq",
        "product_line": "policy",
        "created_at": "2025-04-01T08:00:00Z",
        "updated_at": "2026-03-05T08:00:00Z",
        "is_active": True,
        "text": """
        Does Cordwell match competitor prices?

        Yes, on identical in stock items from a local retail competitor or a major
        online retailer. Bring the advertisement or a link to the pickup desk. Price
        match is applied at the time of purchase or within thirty days after.

        Exclusions are clearance, open box, refurbished, auction, membership only
        pricing, bundle pricing, rebate adjusted pricing, and items sold by third
        party sellers on a marketplace.
        """,
    },
    {
        "document_id": "faq_install_scheduling",
        "source": "faq_installation_scheduling.md",
        "doc_type": "faq",
        "product_line": "policy",
        "created_at": "2025-05-15T08:00:00Z",
        "updated_at": "2026-04-02T08:00:00Z",
        "is_active": True,
        "text": """
        How do I schedule Cordwell Pro Install?

        Purchase the installation service with the product or add it later from
        your order history. A licensed installer contacts you within two business
        days to schedule a measure visit where required.

        Installations performed by Cordwell Pro Install carry a two year labor
        warranty in addition to the product warranty, and labor for a warranty
        removal and reinstallation is covered. Self installed products do not carry
        labor coverage.
        """,
    },
    {
        "document_id": "faq_app_account",
        "source": "faq_app_account.md",
        "doc_type": "faq",
        "product_line": "platform",
        "created_at": "2025-06-20T08:00:00Z",
        "updated_at": "2026-05-11T08:00:00Z",
        "is_active": True,
        "text": """
        Can more than one person control a Cordwell device?

        Yes. The device owner invites additional members from the Cordwell Home app
        under home settings. Members can be given full control or limited control.
        Limited control can operate a device but cannot change settings, remove the
        device, or view the audit history.

        There is no limit on members. Removing a member revokes their access
        immediately on devices that are online and at next connection on devices
        that are offline.
        """,
    },
    {
        "document_id": "faq_recycling",
        "source": "faq_recycling.md",
        "doc_type": "faq",
        "product_line": "policy",
        "created_at": "2025-07-11T08:00:00Z",
        "updated_at": "2025-10-14T08:00:00Z",
        "is_active": True,
        "text": """
        Does Cordwell recycle old batteries and light bulbs?

        Yes. Every Cordwell store has a drop off station near the entrance for
        rechargeable batteries, compact fluorescent bulbs, and light emitting diode
        bulbs. There is no charge and no purchase required.

        We do not accept alkaline batteries, which most jurisdictions allow in
        household waste, or damaged lithium ion batteries, which must go to a
        household hazardous waste facility.
        """,
    },
    {
        "document_id": "faq_gift_cards",
        "source": "faq_gift_cards.md",
        "doc_type": "faq",
        "product_line": "policy",
        "created_at": "2025-02-28T08:00:00Z",
        "updated_at": "2025-08-05T08:00:00Z",
        "is_active": True,
        "text": """
        Do Cordwell gift cards expire?

        No. Cordwell gift cards do not expire and carry no inactivity fee. They can
        be used in store, online, and at the pro desk. A lost card can be replaced
        for its remaining balance with the original receipt.

        Gift cards cannot be redeemed for cash except where required by law, and
        cannot be used to pay a Cordwell Pro account invoice.
        """,
    },
]

# A deliberately stale, deactivated document. It exists so the metadata filter in
# Part F has something real to exclude, and so the staleness discussion in the
# deck has a concrete artifact in the corpus.
_STALE: list[dict] = [
    {
        "document_id": "ts_thermostat_wifi_v1",
        "source": "troubleshoot_thermostat_wifi_LEGACY.md",
        "doc_type": "troubleshooting",
        "product_line": "climate",
        "created_at": "2024-03-01T10:00:00Z",
        "updated_at": "2024-06-15T10:00:00Z",
        "is_active": False,
        "text": """
        Thermostat drops off the network after a power outage, legacy revision

        This revision describes the discontinued ThermaLink T20 and is retained for
        reference only. It is superseded and must not be given to customers.

        Symptom. The thermostat is unreachable from the app after an outage.

        Cause. The T20 rejoins the network automatically after power is restored.
        A thermostat that does not rejoin has a failed radio module.

        Resolution. Replace the unit under warranty. There is no field repair for
        the radio module and there is no re-pairing procedure, because the T20 does
        not have one.

        Note added by support. Applying this article to a ThermaLink T40 produces
        unnecessary unit replacements. The T40 requires manual re-pairing after
        every power interruption by design. Confirm the model before using this
        article.
        """,
    },
]


def _dedent(text: str) -> str:
    """Collapse the indentation used for readability in the literals above."""
    lines = [ln.strip() for ln in text.strip().splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln == "":
            out.append("")
        else:
            out.append(ln)
    # Rejoin, preserving blank lines as paragraph separators.
    paragraphs: list[str] = []
    current: list[str] = []
    for ln in out:
        if ln == "":
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(ln)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def build_documents() -> list[dict]:
    """
    Assemble the full corpus.

    Returns a list of dicts, each with:
        document_id, source, doc_type, product_line,
        created_at, updated_at, is_active, text

    The text of a manual is its sections joined with markdown headings, which is
    what gives the paragraph-aware chunker something real to split on.
    """
    docs: list[dict] = []

    # Real installation manuals are mostly boilerplate. Each manual carries an
    # additional run of appendix sections so that a 1,024 token chunk genuinely
    # mixes an answer-bearing section with unrelated material, which is the
    # dilution the module is about. Fixed order, no randomness in the selection.
    _APPENDIX_ORDER = [
        ("spec", "Extended specifications"),
        ("torque", "Fastener reference, metric"),
        ("safety", "Safety information, restated for service technicians"),
        ("maintenance", "Seasonal service schedule"),
        ("parts", "Accessory and service parts"),
        ("compliance", "Additional regulatory notices"),
        ("warranty", "Warranty claim procedure"),
        ("spec", "Performance data tables"),
        ("maintenance", "Extended service intervals"),
        ("torque", "Fastener reference, imperial"),
    ]

    for m in _MANUALS:
        rng = random.Random(m["seed"])
        parts: list[str] = [f"# {m['product']} installation and service manual"]
        for heading, body in list(m["layout"]) + [
            (f"APPENDIX::{title}", kind) for kind, title in _APPENDIX_ORDER
        ]:
            if heading.startswith("APPENDIX::"):
                title = heading.removeprefix("APPENDIX::")
                text = _boilerplate(body, rng, m["product"], m["pn"], m["model"])
            elif heading == "boiler":
                kind = body
                title = {
                    "safety": "Safety information",
                    "torque": "Fastener torque specifications",
                    "parts": "Carton contents and replacement parts",
                    "warranty": "Limited warranty",
                    "compliance": "Regulatory compliance and ratings",
                    "spec": "Technical specifications",
                    "maintenance": "Scheduled maintenance",
                }[kind]
                text = _boilerplate(kind, rng, m["product"], m["pn"], m["model"])
            else:
                title = heading.removeprefix("ANSWER ")
                text = body
            parts.append(f"## {title}\n\n{_dedent(text)}")
        docs.append(
            {
                "document_id": m["document_id"],
                "source": m["source"],
                "doc_type": m["doc_type"],
                "product_line": m["product_line"],
                "created_at": m["created_at"],
                "updated_at": m["updated_at"],
                "is_active": m["is_active"],
                "text": "\n\n".join(parts),
            }
        )

    for group in (_TROUBLESHOOTING, _FAQS, _STALE, _STALE_EXTRA):
        for d in group:
            docs.append(
                {
                    "document_id": d["document_id"],
                    "source": d["source"],
                    "doc_type": d["doc_type"],
                    "product_line": d["product_line"],
                    "created_at": d["created_at"],
                    "updated_at": d["updated_at"],
                    "is_active": d["is_active"],
                    "text": _dedent(d["text"]),
                }
            )

    docs.extend(_distractor_documents())
    docs.sort(key=lambda d: d["document_id"])
    return docs


# ---------------------------------------------------------------------------
# Distractor documents.
#
# A support knowledge base with twenty five articles is not a retrieval problem.
# Real Cordwell support carries hundreds. The documents below are the rest of the
# catalogue: other products, other symptoms, written in the same house voice.
#
# They are deliberately LEXICALLY CLOSE and SEMANTICALLY DISTINCT. A document
# about the ThermaLink T30 display going blank shares the words thermostat,
# power, app, and network with query q01, and is genuinely not the answer to it.
# That is what makes the ranking task real instead of decorative.
#
# If a distractor were actually a correct answer to a labeled query, the label
# would be wrong and every number in this lab would be meaningless. The catalogue
# is kept to symptoms that no labeled query asks about.
# ---------------------------------------------------------------------------

_CATALOG: list[tuple[str, str, str]] = [
    ("ThermaLink T30", "climate", "thermostat"),
    ("ThermaLink T55", "climate", "thermostat"),
    ("ClimaVent V5", "climate", "vent damper"),
    ("ClimaVent V9", "climate", "vent damper"),
    ("AirSense A2", "climate", "air quality monitor"),
    ("AquaSurge TX5", "plumbing", "tankless water heater"),
    ("AquaSurge TX12", "plumbing", "tankless water heater"),
    ("PureFlow RO6", "plumbing", "reverse osmosis system"),
    ("StormShield Pro", "plumbing", "sump pump"),
    ("LeakSentry LS2", "plumbing", "water leak detector"),
    ("GateWay G4", "access", "garage door opener"),
    ("GateWay G9", "access", "garage door opener"),
    ("BoltGuard L1", "access", "smart deadbolt"),
    ("BoltGuard L5", "access", "smart lock"),
    ("PorchView C3", "access", "video doorbell"),
    ("TurfMate RM1", "outdoor", "robotic mower"),
    ("TurfMate RM7", "outdoor", "robotic mower"),
    ("LumaPath Pro", "outdoor", "landscape lighting system"),
    ("HydroReach HR4", "outdoor", "irrigation controller"),
    ("BlastMate PW3", "outdoor", "pressure washer"),
    ("VoltNest EV2", "electrical", "vehicle charger"),
    ("VoltNest B10", "electrical", "home battery"),
    ("CircuitWatch CW1", "electrical", "energy monitor"),
    ("BrightSet D4", "electrical", "smart dimmer"),
    ("SafeSense S8", "electrical", "smoke and carbon monoxide alarm"),
]

_DISTRACTOR_ISSUES: list[tuple[str, str]] = [
    (
        "display is blank or unresponsive",
        "The {kind} display is dark. Confirm supply power at the breaker. Confirm "
        "the ribbon connector at the control board is fully seated. A display that "
        "flickers and then goes dark under load is a supply problem rather than a "
        "display fault. The unit continues to run its stored program with the "
        "display dark, so do not assume the {kind} has stopped working. Replace the "
        "display assembly only after confirming supply voltage at the board while "
        "the unit is under load. A display that is dim rather than dark in cold "
        "weather is normal for liquid crystal panels and recovers as the housing "
        "warms.",
    ),
    (
        "firmware update fails partway through",
        "A firmware update on the {kind} that stops partway leaves the unit on the "
        "previous version rather than in a broken state. Retry on a stable network "
        "with the phone close to the unit. Updates are delivered through the "
        "Cordwell Home app and require the unit to stay awake for the duration. Do "
        "not power cycle the {kind} during an update. If three attempts fail, "
        "export the diagnostic bundle from the app and escalate to Cordwell "
        "technical support with the bundle attached. Do not attempt to load "
        "firmware from a file, which is not supported on any Cordwell product.",
    ),
    (
        "app shows the wrong state",
        "The Cordwell Home app shows a state for the {kind} that does not match "
        "what the unit is actually doing. This is a reporting problem rather than a "
        "control problem, and the unit is behaving correctly. Force a state refresh "
        "by pulling down on the device card. Confirm the unit clock is "
        "synchronised, because a drifted clock causes the app to present a stale "
        "cached state as current. A unit that has been offline reports its last "
        "known state until it reconnects, which is by design and is not a fault.",
    ),
    (
        "unit is noisy during operation",
        "Unusual noise from the {kind} is diagnosed by when it occurs rather than "
        "by how it sounds. Noise at startup only points to a mounting or bearing "
        "issue. Noise that rises with load points to an obstruction in the flow "
        "path. Noise that is constant regardless of load points to a loose housing "
        "panel or an unsecured cable slapping the enclosure. Confirm all fasteners "
        "are at the specified torque before disassembling anything, because a loose "
        "housing accounts for most reported noise complaints.",
    ),
    (
        "scheduled program does not run",
        "The {kind} schedule did not run as configured. Check whether a manual hold "
        "or override is active, which suspends the schedule indefinitely until it "
        "is cleared. Check the unit clock and the configured time zone. Check "
        "whether the schedule was edited from a second account, because the last "
        "edit wins and there is no merge between accounts. Schedules are stored on "
        "the unit and continue to run with the network unavailable, so a missed "
        "schedule is not explained by an outage.",
    ),
    (
        "intermittent connection drops",
        "The {kind} drops off the Cordwell Home app for minutes at a time and "
        "returns without intervention. Read the signal strength in device "
        "diagnostics before changing anything else. Below negative seventy five "
        "decibel milliwatts, expect exactly this pattern. Relocating the access "
        "point or adding a mesh node resolves most cases. A unit that drops at the "
        "same time every day is usually seeing interference from a scheduled "
        "appliance rather than a network fault, so ask what runs on a timer.",
    ),
    (
        "physical installation does not fit the opening",
        "The {kind} does not fit the existing opening. Confirm the rough opening "
        "dimensions against the specification section before modifying anything "
        "structural. Cordwell supplies adapter and extension kits for most common "
        "mismatches, and fitting one is always preferable to enlarging an opening. "
        "Do not modify the housing or the mounting plate to force a fit. A modified "
        "housing voids the limited warranty and can compromise the ingress "
        "protection rating.",
    ),
    (
        "water or moisture inside the housing",
        "Moisture inside the {kind} housing indicates a failed gasket or an "
        "incorrect mounting orientation. Confirm the drip loop on the supply cable. "
        "Confirm the unit is mounted with the cable entry facing down. Replace the "
        "gasket rather than sealing the housing with silicone, which traps "
        "condensation inside and makes the problem permanent. Inspect the board for "
        "corrosion at the terminals before returning the unit to service.",
    ),
]

_DISTRACTOR_FAQ_TOPICS: list[tuple[str, str]] = [
    (
        "what is covered by the {pname} extended protection plan",
        "The extended protection plan covers mechanical and electrical failure "
        "after the limited warranty expires, including labour and parts. It does "
        "not cover cosmetic damage, consumable items, or damage from improper "
        "installation. Plans are purchased within thirty days of the product and "
        "are transferable once.",
    ),
    (
        "how do I find the serial number on the {pname}",
        "The serial number is printed on the product label. On most Cordwell "
        "products the label is inside the battery compartment or behind the cover "
        "plate. The serial number is also shown in the Cordwell Home app under "
        "device details once the product is added to a home.",
    ),
    (
        "is the {pname} compatible with older Cordwell accessories",
        "Accessories from the previous product generation are supported where the "
        "physical interface is unchanged. Radio accessories are not "
        "cross-compatible between generations because the accessory radio protocol "
        "changed. Check the compatibility table on the product page before "
        "purchasing an accessory for an older unit.",
    ),
]

# Additional superseded revisions. These exist so the is_active metadata filter in
# Part F has real, dangerous content to exclude rather than a token example.
_STALE_EXTRA: list[dict] = [
    {
        "document_id": "ts_water_heater_codes_v1",
        "source": "troubleshoot_water_heater_codes_LEGACY.md",
        "doc_type": "troubleshooting",
        "product_line": "plumbing",
        "created_at": "2024-02-10T10:00:00Z",
        "updated_at": "2024-08-01T10:00:00Z",
        "is_active": False,
        "text": """
        AquaSurge fault code reference, legacy revision

        This revision covers the discontinued AquaSurge TX3 and is retained for
        reference only. It is superseded and must not be given to customers.

        E-4471 on the TX3 indicates a blocked condensate drain, not a flame sense
        failure. Clear the drain trap and reset at the panel.

        E-5510 on the TX3 indicates an outlet sensor fault. Replace the sensor.

        Note added by support. The TX3 and TX9 code tables do not match. Applying
        this article to a TX9 sends the technician to the condensate drain for a
        flame sense problem and to a sensor replacement for a scale problem. Both
        are wrong. Confirm the model before using this article.
        """,
    },
    {
        "document_id": "ts_mower_boundary_v1",
        "source": "troubleshoot_mower_boundary_LEGACY.md",
        "doc_type": "troubleshooting",
        "product_line": "outdoor",
        "created_at": "2024-04-05T10:00:00Z",
        "updated_at": "2024-09-12T10:00:00Z",
        "is_active": False,
        "text": """
        TurfMate boundary wire fault, legacy revision

        This revision covers the discontinued TurfMate RM0 and is retained for
        reference only. It is superseded and must not be given to customers.

        The RM0 uses satellite positioning and a stored map rather than a boundary
        wire. A boundary fault on the RM0 is corrected by clearing the stored map
        and walking a new perimeter from the app.

        Note added by support. The RM3 has no stored map and no satellite receiver.
        Telling an RM3 owner to clear the map and walk a perimeter wastes the call
        and does not touch the actual loop continuity problem.
        """,
    },
]


def _distractor_documents() -> list[dict]:
    """
    Build the rest of the catalogue.

    Deterministic: driven entirely by the fixed lists above and a seeded
    Random, never by wall clock or process state.
    """
    rng = random.Random(9021)
    out: list[dict] = []
    for pname, line, kind in _CATALOG:
        slug = pname.lower().replace(" ", "_")
        picks = rng.sample(_DISTRACTOR_ISSUES, 6)
        for j, (title, body) in enumerate(picks):
            out.append(
                {
                    "document_id": f"ts_{slug}_{j}",
                    "source": f"troubleshoot_{slug}_{j}.md",
                    "doc_type": "troubleshooting",
                    "product_line": line,
                    "created_at": "2025-06-01T10:00:00Z",
                    "updated_at": "2026-01-15T10:00:00Z",
                    "is_active": True,
                    "text": _dedent(
                        f"Cordwell {pname} {kind}: {title}\n\n" + body.format(kind=kind)
                    ),
                }
            )
        qtitle, qbody = rng.choice(_DISTRACTOR_FAQ_TOPICS)
        out.append(
            {
                "document_id": f"faq_{slug}",
                "source": f"faq_{slug}.md",
                "doc_type": "faq",
                "product_line": line,
                "created_at": "2025-06-01T08:00:00Z",
                "updated_at": "2026-01-15T08:00:00Z",
                "is_active": True,
                "text": _dedent(
                    qtitle.format(pname=pname) + "\n\n" + qbody.format(pname=pname)
                ),
            }
        )
    return out


# ---------------------------------------------------------------------------
# The labeled query set.
#
# Each entry is a real support question paired with the set of document_ids a
# human judged relevant. This is the human labor the deck refuses to hand wave
# away. Thirty queries is at the low end of the fifty to one hundred the deck
# recommends, sized so the lab finishes inside its slot.
#
# "relevant" holds document_ids, not chunk ids. Chunk hits are deduplicated up
# to their document before scoring.
# ---------------------------------------------------------------------------

LABELED_QUERIES: list[dict] = [
    {
        "query_id": "q01",
        "text": "why does the smart thermostat lose wifi after a power cut",
        "relevant": {"ts_thermostat_wifi", "man_thermostat_t40"},
    },
    {
        "query_id": "q02",
        "text": "how long do I have to re-pair the thermostat before the setup window closes",
        "relevant": {"man_thermostat_t40", "ts_thermostat_wifi"},
    },
    {
        "query_id": "q03",
        "text": "thermostat will not join the network because the phone is on the wrong band",
        "relevant": {"ts_thermostat_wifi", "man_thermostat_t40", "ts_app_pairing"},
    },
    {
        "query_id": "q04",
        "text": "does the thermostat need a common wire for continuous power",
        "relevant": {"man_thermostat_t40"},
    },
    {
        "query_id": "q05",
        "text": "thermostat reads several degrees off from the actual room temperature",
        "relevant": {"man_thermostat_t40"},
    },
    {
        "query_id": "q06",
        "text": "what does fault code E-4471 mean on the tankless water heater",
        "relevant": {"ts_water_heater_codes", "man_water_heater_tx9"},
    },
    {
        "query_id": "q07",
        "text": "water heater lights and then shuts down on the safety timer",
        "relevant": {"ts_water_heater_codes", "man_water_heater_tx9"},
    },
    {
        "query_id": "q08",
        "text": "how often should a tankless heater be descaled in hard water",
        "relevant": {"man_water_heater_tx9"},
    },
    {
        "query_id": "q09",
        "text": "hot water goes cold at high flow in the winter",
        "relevant": {"man_water_heater_tx9"},
    },
    {
        "query_id": "q10",
        "text": "no hot water at one sink only but fine everywhere else",
        "relevant": {"ts_water_heater_codes"},
    },
    {
        "query_id": "q11",
        "text": "garage remote stopped working but the wall button still does",
        "relevant": {"ts_garage_remote", "man_garage_g7"},
    },
    {
        "query_id": "q12",
        "text": "how do I program a new keypad code to the garage opener",
        "relevant": {"man_garage_g7", "ts_garage_remote"},
    },
    {
        "query_id": "q13",
        "text": "garage door goes down partway and then goes back up",
        "relevant": {"man_garage_g7", "ts_garage_remote"},
    },
    {
        "query_id": "q14",
        "text": "how high off the floor should the garage safety eyes be mounted",
        "relevant": {"man_garage_g7"},
    },
    {
        "query_id": "q15",
        "text": "robot mower says it cannot find the boundary wire",
        "relevant": {"ts_mower_boundary", "man_mower_rm3"},
    },
    {
        "query_id": "q16",
        "text": "how do I find a break in the buried perimeter wire",
        "relevant": {"ts_mower_boundary", "man_mower_rm3"},
    },
    {
        "query_id": "q17",
        "text": "how often should the mower blades be changed",
        "relevant": {"man_mower_rm3"},
    },
    {
        "query_id": "q18",
        "text": "mower leaves an uneven cut even with new blades",
        "relevant": {"man_mower_rm3"},
    },
    {
        "query_id": "q19",
        "text": "smart lock batteries only last a couple of months",
        "relevant": {"ts_deadbolt_battery", "man_deadbolt_l2"},
    },
    {
        "query_id": "q20",
        "text": "can I use lithium batteries in the smart deadbolt",
        "relevant": {"man_deadbolt_l2", "ts_deadbolt_battery"},
    },
    {
        "query_id": "q21",
        "text": "how do I get in if the deadbolt battery is completely dead",
        "relevant": {"man_deadbolt_l2"},
    },
    {
        "query_id": "q22",
        "text": "guest code stopped working at the edge of its scheduled window",
        "relevant": {"man_deadbolt_l2"},
    },
    {
        "query_id": "q23",
        "text": "sump pump turns on and off every few seconds",
        "relevant": {"ts_sump_cycling", "man_sump_stormshield"},
    },
    {
        "query_id": "q24",
        "text": "pump runs continuously but no water leaves the pit",
        "relevant": {"ts_sump_cycling", "man_sump_stormshield"},
    },
    {
        "query_id": "q25",
        "text": "landscape lights are dimmer at the far end of the run",
        "relevant": {"ts_lighting_dim"},
    },
    {
        "query_id": "q26",
        "text": "what wire gauge fixes voltage drop on low voltage outdoor lighting",
        "relevant": {"ts_lighting_dim"},
    },
    {
        "query_id": "q27",
        "text": "the app never shows the device during setup",
        "relevant": {"ts_app_pairing"},
    },
    {
        "query_id": "q28",
        "text": "can my spouse control the same devices from their own phone",
        "relevant": {"faq_app_account"},
    },
    {
        "query_id": "q29",
        "text": "how many days do I have to return a purchase",
        "relevant": {"faq_returns"},
    },
    {
        "query_id": "q30",
        "text": "do I have to register the product to keep the warranty",
        "relevant": {"faq_warranty_registration"},
    },
]
