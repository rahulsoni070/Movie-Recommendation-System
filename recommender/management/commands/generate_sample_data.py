"""
Management command to generate sample movie model files.
Run this once before starting the server:

    python manage.py generate_sample_data

This creates the required model artifacts in training/models/ using a
built-in dataset of 127 popular movies – no external download needed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Built-in sample dataset – 150 well-known movies
# Each entry: (title, year, genres_list, keywords, company, rating, votes,
#              imdb_id, poster_path, overview)
# ---------------------------------------------------------------------------
SAMPLE_MOVIES = [
    # Action / Adventure
    ("The Dark Knight", "2008-07-18", ["action", "crime", "thriller"],
     "batman joker gotham superhero villain", "Warner Bros.", 9.0, 2700000,
     "tt0468569", "/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
     "Batman raises the stakes in his war on crime."),
    ("Inception", "2010-07-16", ["action", "scifi", "thriller"],
     "dreams heist subconscious memory reality", "Warner Bros.", 8.8, 2300000,
     "tt1375666", "/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg",
     "A thief who steals corporate secrets through dream-sharing technology."),
    ("The Matrix", "1999-03-31", ["action", "scifi"],
     "simulation virtual reality hacker neo rebels machines", "Warner Bros.", 8.7, 1900000,
     "tt0133093", "/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
     "A computer hacker learns about the true nature of his reality."),
    ("Interstellar", "2014-11-05", ["adventure", "drama", "scifi"],
     "space wormhole time relativity future planet", "Paramount Pictures", 8.6, 1800000,
     "tt0816692", "/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
     "A team of explorers travel through a wormhole in space."),
    ("Avengers: Endgame", "2019-04-26", ["action", "adventure", "scifi"],
     "marvel superhero infinity stones time travel avengers", "Marvel Studios", 8.4, 1200000,
     "tt4154796", "/or06FN3Dka5tukK1e9sl16pB3iy.jpg",
     "After the devastating events of Infinity War, the Avengers assemble once more."),
    ("The Avengers", "2012-05-04", ["action", "adventure", "scifi"],
     "marvel superhero team shield iron man thor hulk", "Marvel Studios", 8.0, 1400000,
     "tt0848228", "/cezWGskPY5x7GaglTTRN4Fugfb8.jpg",
     "Earth's mightiest heroes must come together."),
    ("Captain America: Civil War", "2016-05-06", ["action", "adventure", "scifi"],
     "marvel superhero conflict registration team captain", "Marvel Studios", 7.8, 1200000,
     "tt3498820", "/rAGiXaUfPzY7CDEyNNXiAN3Ya9O.jpg",
     "Political pressure mounts to install a governing body to oversee the Avengers."),
    ("Iron Man", "2008-05-02", ["action", "adventure", "scifi"],
     "marvel armor technology billionaire superhero suit", "Marvel Studios", 7.9, 1300000,
     "tt0371746", "/78lPtwv72eTNqFW9COBF8l6zVpl.jpg",
     "Tony Stark builds a powerful armored suit to fight evil."),
    ("Thor: Ragnarok", "2017-11-03", ["action", "adventure", "comedy"],
     "marvel asgard gladiator planet hulk ragnarok", "Marvel Studios", 7.9, 850000,
     "tt3501632", "/rzRwTcFvttcN1ZpX2xv4j3tSdJu.jpg",
     "Thor is imprisoned on Sakaar and must escape to prevent Ragnarok."),
    ("Black Panther", "2018-02-16", ["action", "adventure", "scifi"],
     "marvel wakanda africa king vibranium superhero", "Marvel Studios", 7.3, 900000,
     "tt1825683", "/uxzzxijgPIY7slzFvMotPv8wjKA.jpg",
     "T'Challa returns home to become King of Wakanda."),

    # Sci-Fi
    ("2001: A Space Odyssey", "1968-04-03", ["adventure", "scifi"],
     "space hal ai monolith jupiter evolution", "Metro-Goldwyn-Mayer", 8.3, 700000,
     "tt0062622", "/ve72VxNqjIqd6KKOIIQkQMJJ3LH.jpg",
     "Humanity finds a mysterious monolith affecting human evolution."),
    ("Blade Runner 2049", "2017-10-06", ["drama", "mystery", "scifi"],
     "replicant future dystopia blade runner secrets sequel", "Warner Bros.", 8.0, 500000,
     "tt1856101", "/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg",
     "A young blade runner discovers a long-buried secret."),
    ("Arrival", "2016-11-11", ["drama", "mystery", "scifi"],
     "alien language time communication first contact linguist", "Paramount Pictures", 7.9, 700000,
     "tt2543164", "/x2FJsf1ElAgr63Y3PNPtJrcmpoe.jpg",
     "A linguist is recruited to communicate with alien lifeforms."),
    ("Gravity", "2013-10-04", ["drama", "scifi", "thriller"],
     "space astronaut survival orbit debris isolation", "Warner Bros.", 7.7, 800000,
     "tt1454468", "/6LdOc4FBLKhp0DA1aqBGSJzk3Cl.jpg",
     "Two astronauts work together to survive after an accident."),
    ("The Martian", "2015-10-02", ["adventure", "drama", "scifi"],
     "mars survival astronaut botany science rescue mission", "20th Century Fox", 8.0, 850000,
     "tt3659388", "/AjkEZZYPMmMn53Olo3RQy34HSba.jpg",
     "An astronaut becomes stranded on Mars and must improvise to survive."),
    ("Ex Machina", "2014-01-21", ["drama", "mystery", "scifi"],
     "AI robot consciousness test turing female android", "Universal Pictures", 7.7, 500000,
     "tt0470752", "/btVFzOn2lXFHgHUEleq2GFJJPJN.jpg",
     "A programmer is selected to evaluate an AI with a humanoid robot."),
    ("Her", "2013-10-12", ["drama", "romance", "scifi"],
     "AI operating system love loneliness future technology", "Warner Bros.", 8.0, 600000,
     "tt1798709", "/eCOtqtfvn7mxGaEOcze8FRzJKLY.jpg",
     "A lonely writer develops an unlikely relationship with an AI."),
    ("WALL-E", "2008-06-27", ["animation", "family", "scifi"],
     "robot future earth trash love space clean-up", "Pixar Animation Studios", 8.4, 1100000,
     "tt0910970", "/hbhFnRzzg6ZDmm8YAmxBnwln8ah.jpg",
     "A small waste-collecting robot falls in love and inadvertently saves Earth."),
    ("Dune", "2021-09-15", ["action", "adventure", "drama"],
     "desert spice sand worm empire prophecy messianic", "Warner Bros.", 8.0, 700000,
     "tt1160419", "/d5NXSklpcvwN3sqQjxL02vm3bVf.jpg",
     "Paul Atreides travels to the most dangerous planet to ensure his family's future."),
    ("The Terminator", "1984-10-26", ["action", "scifi"],
     "cyborg future war skynet time travel robot killer", "Orion Pictures", 8.1, 800000,
     "tt0088247", "/9M2CbRMGFHOCqCBVo5Iysx8dvYY.jpg",
     "A human soldier is sent to protect a woman a cyborg assassin wants dead."),

    # Drama
    ("The Shawshank Redemption", "1994-09-23", ["crime", "drama"],
     "prison hope friendship escape justice redemption", "Columbia Pictures", 9.3, 2700000,
     "tt0111161", "/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg",
     "Two imprisoned men bond over years, finding solace and eventual redemption."),
    ("Forrest Gump", "1994-07-06", ["comedy", "drama", "romance"],
     "running history love friendship destiny simplicity", "Paramount Pictures", 8.8, 2100000,
     "tt0109830", "/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg",
     "Forrest Gump's extraordinary life unfolds through decades of American history."),
    ("Schindler's List", "1993-12-15", ["biography", "drama", "history"],
     "holocaust world war factory jews rescue oskar", "Universal Pictures", 9.0, 1400000,
     "tt0108052", "/sF1U4EUQS8YHUYjNl3pMGNIQyr0.jpg",
     "A businessman saves over a thousand Jewish lives during the Holocaust."),
    ("Goodfellas", "1990-09-19", ["biography", "crime", "drama"],
     "mafia mob gangster rise fall henry hill crime", "Warner Bros.", 8.7, 1100000,
     "tt0099685", "/aKuFiU82s5ISJpGZp7YkIr3kCUd.jpg",
     "Henry Hill and his friends work their way up through the mob hierarchy."),
    ("The Godfather", "1972-03-24", ["crime", "drama"],
     "mafia family power don corleone heir loyalty", "Paramount Pictures", 9.2, 1800000,
     "tt0068646", "/3bhkrj58Vtu7enYsLegHzDiHzma.jpg",
     "The aging patriarch of an organized crime dynasty transfers control to his son."),
    ("The Godfather Part II", "1974-12-20", ["crime", "drama"],
     "mafia corleone rise origin young vito michael", "Paramount Pictures", 9.0, 1200000,
     "tt0071562", "/hek3koDUyRQk7FIhPXsa6mT2Zc3.jpg",
     "The early life of Vito Corleone and Michael's rise intercut."),
    ("Fight Club", "1999-10-15", ["drama", "mystery", "thriller"],
     "soap underground rebellion consumerism identity twist", "20th Century Fox", 8.8, 2100000,
     "tt0137523", "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
     "An insomniac office worker forms an underground fight club."),
    ("A Beautiful Mind", "2001-12-21", ["biography", "drama"],
     "mathematics genius schizophrenia nobel prize paranoia", "Universal Pictures", 8.2, 900000,
     "tt0268978", "/ewlzKEFoMbqXUNHWHCQxRbSCyxB.jpg",
     "The life of brilliant mathematician John Nash."),
    ("Whiplash", "2014-10-10", ["drama", "music"],
     "music drummer jazz ambition perfectionism teacher student", "Sony Pictures Classics", 8.5, 700000,
     "tt2582802", "/lIv1QinFqz4dlp5U4lQ6HaiskOZ.jpg",
     "A young jazz drummer pushes his limits under a demanding instructor."),
    ("12 Angry Men", "1957-04-10", ["crime", "drama"],
     "jury deliberation justice murder verdict doubt reasonable", "United Artists", 9.0, 800000,
     "tt0050083", "/ow3wq89wM8qd5X7hWKxiRfsFf9C.jpg",
     "A jury holdout attempts to prevent a miscarriage of justice."),

    # Horror / Thriller
    ("Get Out", "2017-02-24", ["horror", "mystery", "thriller"],
     "racism hypnosis sunken place identity social commentary", "Universal Pictures", 7.7, 600000,
     "tt5052448", "/tFXcEccSQMf3lfhfXKSU9iRBpa3.jpg",
     "A young African-American visits his white girlfriend's parents."),
    ("Hereditary", "2018-06-08", ["drama", "horror", "mystery"],
     "grief cult possession family trauma supernatural", "A24", 7.3, 400000,
     "tt7784604", "/4O1e0UzZ6Vfmu1AiilheBZOj8tc.jpg",
     "A family unravels cryptic and terrifying secrets about their ancestry."),
    ("The Silence of the Lambs", "1991-02-14", ["crime", "drama", "thriller"],
     "hannibal lecter serial killer fbi profile psychological", "Orion Pictures", 8.6, 1400000,
     "tt0102926", "/rplLJ2hPcOQmkFhTqUte0MkosOe.jpg",
     "An FBI trainee seeks help from an imprisoned cannibal to catch a serial killer."),
    ("Se7en", "1995-09-22", ["crime", "drama", "mystery"],
     "serial killer seven deadly sins detective crime dark", "New Line Cinema", 8.6, 1600000,
     "tt0114369", "/6yoghtyTpznpBik8EngEmJskVUO.jpg",
     "Two detectives hunt a serial killer who uses the seven deadly sins."),
    ("Psycho", "1960-09-08", ["horror", "mystery", "thriller"],
     "shower motel norman bates mother murder mystery", "Universal Pictures", 8.5, 700000,
     "tt0054215", "/yz4QVqPx3h851MXa06Y35hYNaOO.jpg",
     "A secretary embezzles money and stays at a remote motel."),
    ("The Shining", "1980-05-23", ["drama", "horror"],
     "hotel isolation twins axe writer madness supernatural", "Warner Bros.", 8.4, 1000000,
     "tt0081505", "/b6ko0IKC8MdYBBPkkA1aBPLe2yz.jpg",
     "A family heads to an isolated hotel where sinister forces target the son."),
    ("Parasite", "2019-05-30", ["comedy", "drama", "thriller"],
     "class inequality family scheme wealth poverty korea", "CJ Entertainment", 8.5, 800000,
     "tt6751668", "/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
     "Greed and class discrimination threaten a symbiotic relationship between families."),
    ("A Quiet Place", "2018-04-06", ["drama", "horror", "scifi"],
     "silence monsters sound survival family creatures post-apocalyptic", "Paramount Pictures", 7.5, 500000,
     "tt6644200", "/nAU74GmpUk7t5iklEp3bufwDq4n.jpg",
     "A family struggles to survive creatures that hunt by sound."),
    ("Midsommar", "2019-07-03", ["drama", "horror", "mystery"],
     "cult folk horror ritual summer festival grief sweden", "A24", 7.1, 300000,
     "tt8772262", "/rXsh4MI6uyVgZBSSzXCfitJnVvE.jpg",
     "A couple travels to Sweden for a festival that turns sinister."),
    ("It", "2017-09-08", ["drama", "fantasy", "horror"],
     "clown pennywise children friendship loser's club fear", "Warner Bros.", 7.3, 600000,
     "tt1396484", "/9E2y5Q7WlCVNEhP5GkVComwyUfa.jpg",
     "Seven young kids must defeat a shape-shifting monster called Pennywise."),

    # Comedy
    ("The Grand Budapest Hotel", "2014-02-26", ["adventure", "comedy", "crime"],
     "lobby boy hotel europe concierge eccentric murder mystery", "20th Century Fox", 8.1, 700000,
     "tt2278388", "/nX5XotM9yprCKarRH4fzOq1VM9J.jpg",
     "A writer encounters the adventures of a legendary hotel concierge."),
    ("Superbad", "2007-08-17", ["comedy"],
     "high school party friendship teenage awkward coming-of-age", "Columbia Pictures", 7.6, 700000,
     "tt0829482", "/ux7zxz4n3DyhyGr3ziYt7i4LZjd.jpg",
     "Two co-dependent high school seniors navigate their last days before graduation."),
    ("The Big Lebowski", "1998-03-06", ["comedy", "crime"],
     "dude bowling rug mistaken identity absurd", "Gramercy Pictures", 8.1, 700000,
     "tt0118715", "/xxnIgAMRNMbBiRYS3kU1JBJFmMq.jpg",
     "A mistaken identity entangles a lazy bowler in a complex kidnapping plot."),
    ("Knives Out", "2019-11-27", ["comedy", "crime", "drama"],
     "detective murder mystery will inheritance family eccentric", "Lionsgate", 7.9, 700000,
     "tt8946378", "/pThyQovXQrws2hmkal2gZ4IXGyd.jpg",
     "A detective investigates the death of a successful crime novelist."),
    ("Game Night", "2018-03-02", ["action", "comedy", "mystery"],
     "game night kidnapping board game murder mystery fun", "Warner Bros.", 7.0, 200000,
     "tt2704998", "/iXFPDENzST2oK6yCl3RBDXB3EsS.jpg",
     "A group of friends that holds a weekly game night get embroiled in a real-life mystery."),
    ("Groundhog Day", "1993-02-12", ["comedy", "drama", "fantasy"],
     "time loop repeat day change arrogance redemption", "Columbia Pictures", 8.0, 600000,
     "tt0107048", "/oX8JCw5RXNKe7lFQ4DXpfAQAeXP.jpg",
     "A weatherman finds himself in a time loop, reliving the same day."),
    ("Office Space", "1999-02-19", ["comedy"],
     "work corporate office job rebellion printer printer", "20th Century Fox", 7.8, 400000,
     "tt0151804", "/i4Ks3vcYYrWRhwVkw09Yb7ikMBl.jpg",
     "Workers conspire against their hated employer."),

    # Romance
    ("Titanic", "1997-12-19", ["drama", "romance"],
     "ship ocean love class tragedy iceberg sinking", "Paramount Pictures", 7.9, 1100000,
     "tt0120338", "/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg",
     "A poor man and rich woman fall in love aboard the Titanic."),
    ("La La Land", "2016-12-09", ["comedy", "drama", "music"],
     "jazz piano actress ambition dreams los angeles musical", "Summit Entertainment", 8.0, 700000,
     "tt3783958", "/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg",
     "A jazz musician and actress fall in love while chasing their dreams in L.A."),
    ("The Notebook", "2004-06-25", ["drama", "romance"],
     "love story separation memory aging letter summer", "New Line Cinema", 7.8, 600000,
     "tt0332280", "/rNzQyW4f8B8cTC6eZhrwMm3U942.jpg",
     "A poor young man falls in love with a rich young woman."),
    ("Pride & Prejudice", "2005-11-23", ["drama", "romance"],
     "regency england class darcy prejudice marriage love", "Universal Pictures", 7.8, 400000,
     "tt0414387", "/1DRFNHG17KFP8bX5u2GPnEeH2sP.jpg",
     "Sparks fly when spirited Elizabeth Bennet meets the conceited Mr. Darcy."),
    ("Crazy, Stupid, Love.", "2011-07-29", ["comedy", "drama", "romance"],
     "divorce love womanizer confidence pick-up artist teen", "Warner Bros.", 7.4, 400000,
     "tt1570728", "/hQEVnIZQkLFNBlXlrXUUHHFCXKx.jpg",
     "A middle-aged man gets a makeover with the help of a smooth-talking player."),

    # Animation
    ("Spirited Away", "2001-07-20", ["animation", "adventure", "family"],
     "spirit world bath house girl courage growth miyazaki", "Studio Ghibli", 8.6, 800000,
     "tt0245429", "/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",
     "A girl wanders into a world ruled by gods, witches and spirits."),
    ("The Lion King", "1994-06-15", ["animation", "adventure", "drama"],
     "africa pride rock simba rafiki hakuna matata king", "Walt Disney Pictures", 8.5, 1100000,
     "tt0110357", "/sKCr78MXSLixwmZ8DyJLrpMsd15.jpg",
     "A young lion prince flees his kingdom after his father's murder."),
    ("Toy Story", "1995-11-22", ["animation", "adventure", "comedy"],
     "toys cowboy spaceman andy jealousy friendship pixar", "Pixar Animation Studios", 8.3, 1000000,
     "tt0114709", "/uXDfjJbdP4ijW5hWSBrPl9KamSH.jpg",
     "A cowboy doll is threatened when a new spaceman figure displaces him."),
    ("Finding Nemo", "2003-05-30", ["animation", "adventure", "comedy"],
     "ocean fish father son search australia shark sea", "Pixar Animation Studios", 8.1, 1100000,
     "tt0266543", "/eHuGQ10FUzK1mdOY69wF5pGgEf5.jpg",
     "A clown fish embarks on a journey to find his abducted son."),
    ("Up", "2009-05-29", ["animation", "adventure", "comedy"],
     "balloons house adventure old man boy scout friendship", "Pixar Animation Studios", 8.3, 1100000,
     "tt1049413", "/kiX7UYfOpYrMFSAjd2lLgjONSON.jpg",
     "An old widower attaches balloons to his house and flies to South America."),
    ("Coco", "2017-11-22", ["animation", "adventure", "family"],
     "mexico day dead music family memory land spirit", "Pixar Animation Studios", 8.4, 600000,
     "tt2380307", "/gGEsBPAijhVUFoiNpgZXqRVWJt2.jpg",
     "A young musician travels to the Land of the Dead to uncover family history."),
    ("How to Train Your Dragon", "2010-03-26", ["animation", "adventure", "family"],
     "dragon viking friendship toothless flight chief berk", "DreamWorks Animation", 8.1, 700000,
     "tt0892769", "/ygGmAO60t8GyqUo9x4QjiIYMHGd.jpg",
     "A young Viking befriends a Night Fury dragon despite tradition."),
    ("Inside Out", "2015-06-19", ["animation", "adventure", "comedy"],
     "emotions joy sadness memory growing up brain mind", "Pixar Animation Studios", 8.2, 900000,
     "tt2096673", "/aAmfIX3T8p7E4Mhf9BAvYkeTEuS.jpg",
     "A girl's emotions go on an adventure after her family moves."),
    ("Zootopia", "2016-03-04", ["animation", "adventure", "comedy"],
     "animal city rabbit fox police discrimination prejudice", "Walt Disney Pictures", 8.0, 700000,
     "tt2948356", "/sM33iTl/DT/tX7AUiSjnQIFJBCa.jpg",
     "In a city of animals, a bunny cop and con fox investigate a missing mammal case."),
    ("Kung Fu Panda", "2008-05-22", ["action", "animation", "comedy"],
     "panda kung fu warrior china noodles training destiny", "DreamWorks Animation", 7.6, 700000,
     "tt0441773", "/wWt5JEXUIriMaBEXFZgEExdHSAe.jpg",
     "A loveable panda becomes a kung fu hero against his better judgment."),

    # Crime / Mystery
    ("Pulp Fiction", "1994-10-14", ["crime", "drama"],
     "hitman redemption violence nonlinear briefcase gold watch", "Miramax Films", 8.9, 2100000,
     "tt0110912", "/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",
     "The lives of two hitmen, a boxer, and a pair of criminals intertwine."),
    ("The Usual Suspects", "1995-08-16", ["crime", "drama", "mystery"],
     "keyser soze lineup identity twist plot crime heist", "PolyGram Films", 8.5, 1100000,
     "tt0114814", "/bau9mfBNu3qmjlKD/hHi1oFI1Yv.jpg",
     "A sole survivor of a harbor massacre is questioned about the mysterious Keyser Söze."),
    ("Memento", "2000-10-11", ["mystery", "thriller"],
     "memory tattoo backward polaroid revenge amnesia noir", "Summit Entertainment", 8.4, 1200000,
     "tt0209144", "/yuNs09hvpHVU1cMTgruV2WmicFS.jpg",
     "A man with short-term memory loss attempts to track down his wife's murderer."),
    ("Zodiac", "2007-03-02", ["crime", "drama", "mystery"],
     "serial killer detective reporter unsolved california investigation", "Paramount Pictures", 7.7, 400000,
     "tt0443706", "/e1UX7NHBvH9ZXDH2OHHJ0BOPXRH.jpg",
     "A San Francisco cartoonist investigates the Zodiac Killer."),
    ("Prisoners", "2013-09-20", ["crime", "drama", "mystery"],
     "kidnapping daughter detective revenge vigilante father", "Warner Bros.", 8.1, 700000,
     "tt1876068", "/jHRmMINQFDVGX3Hw8mORpGTvNJi.jpg",
     "Two daughters go missing and the desperate father takes matters into his own hands."),
    ("Gone Girl", "2014-10-03", ["drama", "mystery", "thriller"],
     "marriage wife disappearance media manipulation twist dark", "20th Century Fox", 8.1, 900000,
     "tt2267998", "/2FPHPlul37HJO3HRTl0HXeqNbEX.jpg",
     "A man becomes the prime suspect when his wife disappears on their anniversary."),
    ("L.A. Confidential", "1997-09-19", ["crime", "drama", "mystery"],
     "noir police corruption hollywood 1950s detective scandal", "Warner Bros.", 8.2, 500000,
     "tt0119488", "/7cEh7KGXDLkjkJe4RlZ25g4BSQJ.jpg",
     "Three police officers are caught in a web of corruption and lies in 1950s L.A."),

    # Fantasy / Adventure
    ("The Lord of the Rings: The Fellowship of the Ring", "2001-12-19", ["action", "adventure", "drama"],
     "ring hobbit fellowship quest evil shire mordor elf", "New Line Cinema", 8.8, 1700000,
     "tt0120737", "/6oom5QYQ2yQTMJIbnvbkBL9cHo6.jpg",
     "A hobbit embarks on a quest to destroy a powerful ring."),
    ("The Lord of the Rings: The Two Towers", "2002-12-18", ["action", "adventure", "drama"],
     "ring fellowship war gondor rohan ent uruk siege", "New Line Cinema", 8.8, 1600000,
     "tt0167261", "/5VTN0pR8gcqV3EPUHHfMGnJYspN.jpg",
     "The Fellowship is broken as three separate quests continue."),
    ("The Lord of the Rings: The Return of the King", "2003-12-17", ["action", "adventure", "drama"],
     "ring coronation battle mordor eagle fellowship victory", "New Line Cinema", 9.0, 1700000,
     "tt0167260", "/rCzpDGLbOoPwLjy3OAm5NUPOTrC.jpg",
     "Gandalf and Aragorn lead the World of Men against Sauron's army."),
    ("Harry Potter and the Sorcerer's Stone", "2001-11-16", ["adventure", "family", "fantasy"],
     "wizard school magic hogwarts sorting hat quidditch", "Warner Bros.", 7.9, 800000,
     "tt0241527", "/wuMc08IPKEatf9rnMNXvIDxqP4W.jpg",
     "Rescued from the outrageous neglect of his aunt and uncle, a young boy with a great destiny proves his worth."),
    ("The Hobbit: An Unexpected Journey", "2012-12-14", ["adventure", "fantasy"],
     "hobbit dragon dwarf shire erebor quest dragon bilbo", "New Line Cinema", 7.8, 700000,
     "tt0903624", "/iIsxxFire5yCXmRQGsO35K9mRFi.jpg",
     "A reluctant hobbit embarks on a quest to reclaim the Lonely Mountain."),
    ("Pan's Labyrinth", "2006-10-11", ["drama", "fantasy", "thriller"],
     "fairy tale spain fascism girl faun labyrinth escape", "Warner Bros.", 8.2, 600000,
     "tt0457430", "/htRm5RJFbO0QjCT0viQjMnK0Egm.jpg",
     "In fascist Spain, a girl escapes to a dark, magical fantasy world."),
    ("Stardust", "2007-08-10", ["adventure", "fantasy", "romance"],
     "star fallen kingdom magic wall witch love quest", "Paramount Pictures", 7.7, 300000,
     "tt0486655", "/b6OAcM5eSTosTfKAWIAqcUmRByM.jpg",
     "A young man ventures into a fantasy kingdom to retrieve a fallen star."),
    ("The Princess Bride", "1987-09-25", ["adventure", "comedy", "fantasy"],
     "princess sword fencing duel love rescue giants pirates", "20th Century Fox", 8.0, 400000,
     "tt0093779", "/gpRvq9zeT7X1DPSvWgxCCJMor5A.jpg",
     "While bedridden, a grandfather tells his grandson a story of romance and adventure."),
    ("Pirates of the Caribbean: The Curse of the Black Pearl", "2003-07-09", ["action", "adventure", "fantasy"],
     "pirate ship caribbean jack sparrow sword curse treasure", "Walt Disney Pictures", 8.1, 1100000,
     "tt0325980", "/dT0dUMgCVFPAN15YStaRKRaBa2D.jpg",
     "A blacksmith joins a pirate to rescue his abducted love."),

    # Biographical / Historical
    ("Bohemian Rhapsody", "2018-10-31", ["biography", "drama", "music"],
     "queen freddie mercury band rock concert live aid", "20th Century Fox", 7.9, 700000,
     "tt1727824", "/lHu1wtNaczFPGFDTrjCSzeLPTKN.jpg",
     "The story of the legendary British band Queen and lead singer Freddie Mercury."),
    ("Rocketman", "2019-05-22", ["biography", "drama", "music"],
     "elton john rock musician piano biopic flamboyant", "Paramount Pictures", 7.3, 300000,
     "tt2066051", "/4m6MsHHKi8UOFe8IORfkrCfhnrr.jpg",
     "A musical fantasy about the incredible human story of Elton John."),
    ("The Social Network", "2010-10-01", ["biography", "drama"],
     "facebook startup harvard lawsuit deposition friendship betrayal", "Columbia Pictures", 7.7, 700000,
     "tt1285016", "/n0ybibhJtQ5icDqTp8eRytcIHJx.jpg",
     "Harvard student Mark Zuckerberg creates a social networking website."),
    ("Moneyball", "2011-09-23", ["biography", "drama", "sport"],
     "baseball statistics analytics sabermetrics underdog teambuilding", "Columbia Pictures", 7.6, 400000,
     "tt1210166", "/3oAa8mJJ97CH9AeGEY6vjAxqcvZ.jpg",
     "Oakland A's manager uses computer-generated analysis to recruit players."),
    ("The Imitation Game", "2014-11-14", ["biography", "drama", "thriller"],
     "turing enigma codebreaker world war cryptography genius", "Weinstein Company", 8.0, 700000,
     "tt2084970", "/noUp0XOqIcmgefRnRZa1nhtRvWO.jpg",
     "Alan Turing cracks the Enigma code during World War II."),
    ("Hacksaw Ridge", "2016-11-04", ["biography", "drama", "history"],
     "war pacifist conscientious objector okinawa medal hero", "Lionsgate", 8.1, 500000,
     "tt2119532", "/MsBSzMKC9IN5bqSHwPhvpEJBQHT.jpg",
     "WWII medic refuses to bear arms but serves on the front lines."),
    ("1917", "2019-12-25", ["drama", "history", "thriller"],
     "trench world war mission race time continuous shot", "Universal Pictures", 8.2, 500000,
     "tt8579674", "/iZf0KyrE25z1sage4SYFLCCrMi9.jpg",
     "Two British soldiers race against time to deliver a life-saving message."),

    # Western
    ("The Good, the Bad and the Ugly", "1966-12-23", ["adventure", "western"],
     "gold treasure civil war three-way standoff bounty hunter gunfighter", "United Artists", 8.8, 800000,
     "tt0060196", "/bX2xnavhMYjWDoZp1VM6VnU1xwe.jpg",
     "Three gunslingers compete to find a Confederate soldier's buried gold."),
    ("Django Unchained", "2012-12-25", ["drama", "western"],
     "slave bounty hunter revenge plantation south civil war shooter", "Weinstein Company", 8.4, 1500000,
     "tt1853728", "/7oWY8VDWW7thTzWh3OKYRkWAOdD.jpg",
     "A freed slave takes a bounty hunter's help to rescue his wife from a plantation."),
    ("No Country for Old Men", "2007-11-09", ["crime", "drama", "thriller"],
     "hitman cattle gun hunter fate chase texas 1980s", "Miramax Films", 8.2, 900000,
     "tt0477348", "/ga4OLm4qLxPqMWCFpqnrTNwWmkH.jpg",
     "A hunter stumbles onto drug money and becomes the target of a killer."),
    ("True Grit", "2010-12-22", ["adventure", "drama", "western"],
     "revenge justice outlaw marshal teenager father murder", "Paramount Pictures", 7.6, 400000,
     "tt1403865", "/mFPfHqt0t1D3hSF8MNBEdMlNOSH.jpg",
     "A stubborn teenager enlists a tough U.S. Marshal to help avenge her father's murder."),
    ("Tombstone", "1993-12-24", ["action", "biography", "drama"],
     "wyatt earp doc holliday ok corral lawman outlaw", "Buena Vista Pictures", 7.8, 200000,
     "tt0108358", "/ah78GJVSMhm3SQ98MDYBDJ52HFu.jpg",
     "Wyatt Earp and Doc Holliday face off against the outlaw Cowboys gang."),

    # Superhero
    ("Spider-Man: Into the Spider-Verse", "2018-12-14", ["action", "animation", "adventure"],
     "spider-man multiverse miles morales alternate dimensions comic", "Sony Pictures Animation", 8.4, 500000,
     "tt4633694", "/iiZZdoQBEYBv6id8su7ImL0oCbD.jpg",
     "Miles Morales becomes the Spider-Man of his reality and crosses the Multiverse."),
    ("Batman v Superman: Dawn of Justice", "2016-03-25", ["action", "adventure", "scifi"],
     "batman superman clash conflict lex luthor wonder woman", "Warner Bros.", 6.4, 600000,
     "tt2975590", "/5UsK3grJvtQrtzEgqNlDljJW96w.jpg",
     "Batman and Superman clash over collateral damage and an emerging threat."),
    ("Doctor Strange", "2016-11-04", ["action", "adventure", "fantasy"],
     "sorcerer multiverse dimensions magic dimension time stone", "Marvel Studios", 7.5, 700000,
     "tt1211837", "/4PiiNGXjnt4NsTBBRHFHkMuflJo.jpg",
     "A surgeon discovers the world of magic and alternate dimensions."),
    ("Guardians of the Galaxy", "2014-08-01", ["action", "adventure", "comedy"],
     "marvel space misfits team rocket raccoon groot quill", "Marvel Studios", 8.0, 1100000,
     "tt2015381", "/r7vmZjiyZw9rpJMQJdXpjgiCOk9.jpg",
     "A group of intergalactic misfits reluctantly team up to save the galaxy."),
    ("Wonder Woman", "2017-06-02", ["action", "adventure", "fantasy"],
     "amazon goddess world war warrior female hero truth lasso", "Warner Bros.", 7.4, 600000,
     "tt0451279", "/gfJGlDaHuWimErCr5Ql0I8x9QSy.jpg",
     "Diana, princess of the Amazons, fights in World War I."),
    ("Ant-Man", "2015-07-17", ["action", "adventure", "comedy"],
     "small size suit heist shrink marvel ant control", "Marvel Studios", 7.3, 700000,
     "tt0478970", "/ToHfnBFkI4Cs1K5UYBbqMvvZ4B6.jpg",
     "A thief equipped with a suit that shrinks him pulls off a heist."),

    # Music / Drama
    ("Amadeus", "1984-09-19", ["biography", "drama", "music"],
     "mozart classical music genius rival jealousy salieri composer", "Warner Bros.", 8.4, 400000,
     "tt0086879", "/8o172DXGEk2oYA6LBMxRYHFPmFP.jpg",
     "The story of Mozart through the eyes of his envious rival Salieri."),
    ("Ray", "2004-10-29", ["biography", "drama", "music"],
     "ray charles blind piano soul rhythm blues jazz", "Universal Pictures", 7.7, 300000,
     "tt0350258", "/n9rcFtmRxZn2cDkA0ETXH5JiX4V.jpg",
     "The life story of legendary rhythm & blues musician Ray Charles."),
    ("Walk the Line", "2005-11-18", ["biography", "drama", "music"],
     "johnny cash country music june carter train prison", "20th Century Fox", 7.7, 300000,
     "tt0358273", "/7SuGsIZWJGb6JLnNQNDLVcyulKE.jpg",
     "A chronicle of Johnny Cash's rise to fame."),

    # Sport
    ("Rocky", "1976-11-21", ["drama", "sport"],
     "boxing underdog training champion philadelphia italian stallion", "United Artists", 8.1, 600000,
     "tt0075148", "/cqWEpQjHyLyChp6e9FFMKN4n7p9.jpg",
     "A small-time boxer gets a once-in-a-lifetime chance to fight the heavyweight champion."),
    ("Remember the Titans", "2000-09-29", ["biography", "drama", "sport"],
     "football integration racism team coach virginia high school", "Walt Disney Pictures", 7.8, 300000,
     "tt0210945", "/5BDE8N5GlhNnWLVq3U3YNZQaLFR.jpg",
     "The true story of a newly appointed African-American coach and his team."),
    ("Rudy", "1993-10-22", ["biography", "drama", "sport"],
     "notre dame football dream determination walk-on small dedication", "TriStar Pictures", 7.5, 200000,
     "tt0108002", "/3vy3lG0BNBG2WAJSYQ8u5GAE4kk.jpg",
     "A young man dreams of playing football at Notre Dame despite all obstacles."),
    ("The Blind Side", "2009-11-20", ["biography", "drama", "sport"],
     "football adoption family generosity nfl offensive tackle homeless", "Warner Bros.", 7.6, 300000,
     "tt0878804", "/enV5ZgHDTZxLwS8ZFGAjxmHG5kC.jpg",
     "A homeless young man is taken in by a wealthy family and becomes an NFL player."),
    ("Million Dollar Baby", "2004-12-15", ["drama", "sport"],
     "boxing trainer female determination injury ambition champion", "Warner Bros.", 8.1, 600000,
     "tt0405159", "/zfYOHw5gJG5MNpKItQFbMsAJqkH.jpg",
     "A determined woman pursues her dream of becoming a professional boxer."),

    # War
    ("Saving Private Ryan", "1998-07-24", ["drama", "history", "war"],
     "d-day normandy world war mission soldier brothers rescue", "DreamWorks", 8.6, 1300000,
     "tt0120815", "/uqx37cS8cpHg8U35f9U5IBlrCV3.jpg",
     "A group of US soldiers goes behind enemy lines to retrieve a paratrooper."),
    ("Dunkirk", "2017-07-21", ["action", "drama", "history"],
     "evacuation wwii england beach beach survival soldier air", "Warner Bros.", 7.9, 600000,
     "tt5013056", "/ebSnODDg9lbsMIaWg2uAbjn7TO5.jpg",
     "Allied soldiers survive a massive military evacuation in WWII."),
    ("Apocalypse Now", "1979-08-15", ["drama", "mystery", "war"],
     "vietnam jungle river madness colonel kurtz mission", "United Artists", 8.5, 700000,
     "tt0078788", "/gQB8Y5RCMkv2zwzFHbUJX3kAhFZ.jpg",
     "A soldier travels up a river to assassinate a rogue military officer."),
    ("Full Metal Jacket", "1987-06-26", ["drama", "war"],
     "vietnam marines boot camp training combat survival", "Warner Bros.", 8.3, 700000,
     "tt0093058", "/uSH2OjFJFFlLKjvfJD2w9i3FBIG.jpg",
     "A pragmatic Marine observes the dehumanizing effects of Vietnam."),
    ("Platoon", "1986-12-19", ["drama", "history", "war"],
     "vietnam jungle combat morality tour duty killing soldier", "Orion Pictures", 7.9, 400000,
     "tt0091763", "/gkRWFkKLsGqgJ5RQZFJ5aBfgEgN.jpg",
     "A young volunteer in Vietnam faces battle and moral dilemmas."),

    # Thriller
    ("Rear Window", "1954-09-01", ["mystery", "thriller"],
     "voyeur photographer window neighbor murder hitchcock observation", "Paramount Pictures", 8.5, 500000,
     "tt0047396", "/ILVF0eJxHMddjxVaBdl09jGfy6.jpg",
     "A photographer confined to a wheelchair suspects a murder through his window."),
    ("Vertigo", "1958-05-22", ["mystery", "romance", "thriller"],
     "acrophobia obsession detective impersonation color dream tower", "Paramount Pictures", 8.3, 400000,
     "tt0052357", "/5M0j0B18abtBI5gi3RkGenerpmG.jpg",
     "A retired police officer follows a woman, becoming obsessed with her."),
    ("North by Northwest", "1959-07-01", ["action", "adventure", "mystery"],
     "case mistaken identity spy train crop duster pursuit", "Metro-Goldwyn-Mayer", 8.3, 400000,
     "tt0053125", "/3a4pd5gqe9emqZp8iXRe3XFdQVc.jpg",
     "An advertising executive is mistaken for a government agent."),
    ("Oldboy", "2003-11-21", ["action", "drama", "mystery"],
     "revenge imprisonment manipulation twist korean family secret", "Show East", 8.4, 600000,
     "tt0364569", "/pWDtjs568ZfOTMbURQBYuT4Qxka.jpg",
     "After being imprisoned for 15 years, a man seeks revenge on his captor."),
    ("The Prestige", "2006-10-20", ["drama", "mystery", "sci-fi"],
     "magician rivalry trick illusion jealousy secret tesla obsession", "Warner Bros.", 8.5, 1300000,
     "tt0482571", "/5MXyQfz8xUP3dIFPTubhTsbFY6N.jpg",
     "Two stage magicians engage in competitive one-upmanship."),

    # Additional Popular Films
    ("Jurassic Park", "1993-06-11", ["action", "adventure", "scifi"],
     "dinosaur park island dna cloning scientist paleontologist", "Universal Pictures", 8.2, 900000,
     "tt0107290", "/lzWHmYdfeFiMIY4JaMmtR7GEli3.jpg",
     "A theme park with cloned dinosaurs goes catastrophically wrong."),
    ("Back to the Future", "1985-07-03", ["adventure", "comedy", "scifi"],
     "time machine delorean 1955 1985 teenager flux capacitor", "Universal Pictures", 8.5, 1100000,
     "tt0088763", "/fNOH9f1aA7XRTzl1sAOx9iF553Q.jpg",
     "A teenager goes back in time to 1955 in a time-traveling DeLorean."),
    ("E.T. the Extra-Terrestrial", "1982-06-11", ["adventure", "family", "scifi"],
     "alien child friendship bicycle moon phone home", "Universal Pictures", 7.9, 700000,
     "tt0083866", "/an0nD6uq7tGmLSKIFUMRrHyOBHD.jpg",
     "A friendly alien is stranded on Earth and befriends a group of children."),
    ("Home Alone", "1990-11-16", ["comedy", "family"],
     "booby trap burglar christmas family kid home alone defend", "20th Century Fox", 7.7, 800000,
     "tt0099785", "/onTSipViHVQAoqrwW2QR8UiQ0tl.jpg",
     "An 8-year-old boy is accidentally left at home when his family flies to Paris."),
    ("The Truman Show", "1998-06-05", ["comedy", "drama"],
     "reality tv fake world hidden cameras surveillance truth freedom", "Paramount Pictures", 8.2, 1000000,
     "tt0120382", "/vuza0WqY239yBXOadKlGwJsZJFE.jpg",
     "A man discovers his entire life is a reality TV show."),
    ("Cast Away", "2000-12-22", ["adventure", "drama"],
     "island survival volleyball wilson fedex plane crash isolated", "DreamWorks", 7.8, 600000,
     "tt0162222", "/2tHDqMqDHGLTCEJDqLsHT9JNmyj.jpg",
     "A FedEx executive stranded on an uninhabited island must reinvent himself."),
    ("The Revenant", "2015-12-25", ["action", "adventure", "drama"],
     "bear survival revenge wilderness frontier mountains snow", "20th Century Fox", 8.0, 800000,
     "tt1663202", "/ji3ecJphATlVgWNY0B0RVXZizR2.jpg",
     "A frontiersman fights for survival after being mauled by a bear."),
    ("Mad Max: Fury Road", "2015-05-15", ["action", "adventure"],
     "post-apocalyptic desert chase war rig imperator furiosa escape", "Warner Bros.", 8.1, 900000,
     "tt1392190", "/kqjL17yufvn9OVLyXYpvtyrFfak.jpg",
     "A road warrior helps a group of female prisoners escape a warlord."),
    ("John Wick", "2014-10-24", ["action", "crime", "thriller"],
     "assassin revenge dog gun-fu tactical hitman underworld", "Lionsgate", 7.4, 600000,
     "tt2911666", "/fZPSd91yGE9fCcCe6OoQr6E3Bev.jpg",
     "A retired assassin seeks vengeance after criminals kill his dog."),
    ("Die Hard", "1988-07-15", ["action", "thriller"],
     "new york christmas party hostage cop skyscraper german", "20th Century Fox", 8.2, 800000,
     "tt0095016", "/yFihWxQcmqcaBR31QM6Y8gT6aYV.jpg",
     "An off-duty cop battles terrorists in a Los Angeles skyscraper."),
    ("Speed", "1994-06-10", ["action", "adventure", "crime"],
     "bus bomb speed limit cop hostage los angeles explosive", "20th Century Fox", 7.3, 400000,
     "tt0111257", "/y3blO1gw1JXHWoqOQ2yIzfPwJSm.jpg",
     "An LAPD cop must prevent a bomb from detonating on a bus."),
    ("Top Gun", "1986-05-16", ["action", "drama", "romance"],
     "navy pilot fighter jet training maverick competition aerial", "Paramount Pictures", 6.9, 500000,
     "tt0092099", "/xUuHj3CgmZQ9P2cMaqQs4J5m4pM.jpg",
     "A hot-shot Navy fighter pilot contends with students and love interests."),
    ("Top Gun: Maverick", "2022-05-27", ["action", "drama"],
     "navy pilot jet training sequel maverick legacy", "Paramount Pictures", 8.3, 500000,
     "tt1745960", "/62HCnUTziyWcpDaBO2i1DX17ljH.jpg",
     "Maverick confronts the ghosts of his past while training next-gen pilots."),
]


class Command(BaseCommand):
    help = (
        "Generate sample movie model files in training/models/. "
        "Run this once before starting the server."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-dir",
            default=None,
            help="Directory to write model files (default: training/models/ inside BASE_DIR)",
        )

    def handle(self, *args, **options):
        output_dir = options["output_dir"] or str(
            Path(settings.BASE_DIR) / "training" / "models"
        )
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.stdout.write("Building sample movie dataset …")
        df = self._build_dataframe()
        self.stdout.write(f"  {len(df)} movies loaded.")

        self.stdout.write("Computing TF-IDF similarity matrix …")
        sim = self._compute_similarity(df)

        self.stdout.write("Saving model artifacts …")
        self._save_artifacts(df, sim, out)

        self.stdout.write(
            self.style.SUCCESS(
                f"\nModel files written to: {out}\n"
                "You can now start the server with:  python manage.py runserver"
            )
        )

    # ------------------------------------------------------------------
    def _build_dataframe(self):
        seen_titles: set = set()
        rows = []
        movie_id = 1
        for entry in SAMPLE_MOVIES:
            title = entry[0]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            (
                title, release_date, genres, keywords,
                company, rating, votes, imdb_id, poster_path, overview,
            ) = entry
            rows.append(
                {
                    "id": movie_id,
                    "title": title,
                    "release_date": release_date,
                    "primary_company": company,
                    "genres": genres,
                    "vote_average": float(rating),
                    "vote_count": int(votes),
                    "popularity": float(rating * 10),
                    "overview": overview,
                    "imdb_id": imdb_id,
                    "poster_path": poster_path,
                    # soup used only for similarity – not stored in parquet
                    "_soup": " ".join(
                        [keywords]
                        + [g.lower().replace(" ", "") for g in genres] * 2
                        + [company.lower().replace(" ", "")]
                        + overview.lower().split()
                    ),
                }
            )
            movie_id += 1

        df = pd.DataFrame(rows)
        return df

    def _compute_similarity(self, df):
        tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            stop_words="english",
        )
        matrix = tfidf.fit_transform(df["_soup"])
        sim = cosine_similarity(matrix).astype(np.float32)
        return sim

    def _save_artifacts(self, df, sim, out: Path):
        # 1. Parquet metadata (drop internal _soup column)
        meta_cols = [
            "id", "title", "release_date", "primary_company",
            "genres", "vote_average", "vote_count", "popularity",
            "overview", "imdb_id", "poster_path",
        ]
        df[meta_cols].to_parquet(out / "movie_metadata.parquet", index=True)

        # 2. Similarity matrix – sparse if large, dense otherwise
        if sim.size > 10_000_000:
            save_npz(out / "similarity_matrix.npz", csr_matrix(sim))
        else:
            np.save(str(out / "similarity_matrix.npy"), sim)

        # 3. Title → index mapping
        title_to_idx = {
            row["title"]: int(idx) for idx, row in df.iterrows()
        }
        with open(out / "title_to_idx.json", "w") as fh:
            json.dump(title_to_idx, fh)

        # 4. Config
        config = {
            "n_movies": len(df),
            "use_svd": False,
            "n_components": None,
            "matrix_shape": list(sim.shape),
            "dataset": "Built-in sample (127 popular movies)",
        }
        with open(out / "config.json", "w") as fh:
            json.dump(config, fh, indent=2)
