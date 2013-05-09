%
% get mfcc coefficients
%

N = 224;                    % let N be the number of files
D = 13;                     % let D be the number of attributes

Attributes = zeros(N,D);    % let Attributes be the attribute matrix. TODO: change to actual attributes

                            % let Songs be a list of the N songs
Songs = {  '01  Holly Valance - Kiss Kiss.mp3';
            '01 - Avril Lavigne - Girlfriend.mp3';
  		'01 - Daughtry - It''s Not Over.mp3';
			'01 - have a nice day.mp3';
			'01-bullet_for_my_valentine-tears_dont_fall_album_edit-mst.mp3';
			'01-evanescence-sweet_sacrifice.mp3';
			'01 - Times Like These [DeadPoetRIP].mp3';
			'01-the_pierces-secret.mp3';
			'02 - 21 Guns.mp3';
			'02 - Do This Anymore.mp3';
			'02 The Bad Touch.mp3';
			'02-ciara_feat_chamillionaire-get_up.mp3';
			'03 - Daughtry - Home.mp3';
			'03 - Mandy Moore - Only Hope.mp3';
			'03 - Open Your Eyes.mp3';
			'03 - Someday.mp3';
			'02 - Everyday.mp3';
			'02 - Learn To Fly [DeadPoetRIP].mp3';
			'02 - Mandy Moore - Someday We''ll Know.mp3';
			'02 - Nickelback - Burn It To The Ground [[-------NEO-THE ONE-------]]    .mp3';
			'03SO FAR AWAY.mp3';
			'04 - Avril Lavigne - The Best Damn Thing.mp3';
			'04 - Burn It Down.mp3';
			'04 - Daughtry - Over you.mp3';
			'04 - Joey.mp3';
			'04 - Mandy Moore - Its Gonna Be Love.mp3';
			'04 - Nickelback - I''d Come For You [[-------NEO-THE ONE-------]]    .mp3';
			'03-nickelback-leader_of_men-eNT.mp3';
			'04-evanescence-lithium.mp3';
			'04-papa_roach-take_me-h8me.mp3';
			'04. Barenaked Ladies - Pinch Me.mp3';
			'05 - Daughtry - Crashed.mp3';
			'05 - Feelin Way Too Damn Good.mp3';
			'05 - Metalingus.mp3';
			'05 - Misunderstood.mp3';
			'05 There She Goes.mp3';
			'05-papa_roach-getting_away_with_murder-h8me.mp3';
			'05-rihanna-shut_up_and_drive-osc.mp3';
			'06 - Because Of You.mp3';
			'06 - Broken Wings.mp3';
			'06 - Daughtry - Feels Like Tonight.mp3';
			'06 - New Radicals - Mother We Just Can''t Get Enough.mp3';
			'06ZOE JANE.mp3';
			'07 - Avril Lavigne - Hot.mp3';
			'07 - Daughtry - What I Want (Featuring Slash).mp3';
			'07 - In Loving Memory.mp3';
			'08 - Blackbird.mp3';
			'08 - Daughtry - Breakdown.mp3';
			'08 - Down to My Last.mp3';
			'08 - Switchfoot - Learning to Breathe.mp3';
			'08-papa_roach-scars-h8me.mp3';
			'08-stone_sour-through_glass_2.mp3';
			'08-With Arms Wide Open-Creed Human Clay.mp3';
			'09 - Daughtry - Gone.mp3';
			'09 - Switchfoot - I Dare You To Move.mp3';
			'09-Higher-Creed Human Clay.mp3';
			'10 - Breaking Benjamin - Unknown Soldier.mp3';
			'10 - Switchfoot- You.mp3';
			'11 - All My Life [DeadPoetRIP].mp3';
			'11 - Daughtry - All These Lives.mp3';
			'11 - Switchfoot - Only Hope.mp3';
			'11 OneRepublic - Stop And Stare.mp3';
			'11 Rooster - Angels Calling.mp3';
			'07 - Toploader - Dancing In The Moonlight.mp3';
			'07-bullet_for_my_valentine-all_these_things_i_hate_(revolve_around_me)-fnt.mp3';
			'12 - Daughtry - What About Now.mp3';
			'12 - Snow Patrol - Chasing Cars      [Torrent-Tatty].mp3';
			'12-dan_finnerty_and_the_dan_band-candy_shop.mp3';
			'12-evanescence-all_that_im_living_for.mp3';
			'16 - THNKS FR TH MMRS - FALL OUT BOY.mp3';
			'17 - 30 Seconds to Mars - The Kill (Bury Me).mp3';
			'25  Britney Spears-Break The Ice.mp3';
			'30 seconds to mars - from yesterday.mp3';
			'31 Timbaland  ft One Republic-Apologize.mp3';
			'6415_3OH3_Kesha-My_F.mp3';
			'A Perfect Circle - Passive.mp3';
			'aari aari.mp3';
			'ACDC - TNT.mp3';
			'aerosmith - dream on.mp3';
			'Aerosmith - Dude Looks Like A Lady.mp3';
			'Aicha.mp3';
			'AKCENT feat RUXANDRA BAR-FEELINGS ON FIRE.mp3';
			'AKCENT feat SHAHZODA-ALL ALONE.mp3';
			'AKCENT-JOKERO.mp3';
			'AKCENT-KING OF DISCO.mp3';
			'AKCENT-KYLIE.mp3';
			'AKCENT-LOVE STONED.mp3';
			'AKCENT-MY PASSION.mp3';
			'AKCENT-STAY WITH ME.mp3';
			'AKCENT-THAT''S MY NAME.mp3';
			'American Pie Soundtrack - Story Of A Girl (featuring Blink 182, Matchbox 20, Third Eye Blind).mp3';
			'Audioslave - Be Yourself.mp3';
			'Audioslave - Doesnt Remind Me.mp3';
			'Avril Lavigne - Nobodys Home.mp3';
			'Away From The Sun.mp3';
			'Backstreet Boys - Incomplete.mp3';
			'Be Somebody.mp3';
			'Bitter Sweet Symphony - The Verve .mp3';
			'BLACK OR WHITE.mp3';
			'Bleed It Out.mp3';
			'Blink 182 - All The Small Things.mp3';
			'Blink 182 - Always.mp3';
			'Blink 182 - Feeling This.mp3';
			'Blink 182 - I Miss You.mp3';
			'Blink 182 - Not Now.mp3';
			'Blink182 - Every Time I Look For You.mp3';
			'Blur - Woo Hoo.mp3';
			'breaking benjamin - rain.mp3';
			'Breaking Benjamin - So Cold.mp3';
			'Breaking Benjamin - The Diary of Jane.mp3';
			'Breaking Benjamin- Hidden Track.mp3';
			'Breaking Benjamins - Away.mp3';
			'breaking the habit.mp3';
			'Chevelle - I Get It.mp3';
			'ChumbaWumba - Tubthumpin.mp3';
			'Coldplay - Fix You.mp3';
			'Coldplay - In My Place.mp3';
			'Coldplay - Speed Of Sound.mp3';
			'Coldplay - Swallowed In The Sea.mp3';
			'Coldplay - The Scientist.mp3';
			'Coldplay - Yellow.mp3';
			'Counting Crows - Accidentaly in love.mp3';
			'Counting Crows - Colorblind.mp3';
			'cowboy.mp3';
			'crawling.mp3';
			'Creed - Six Feet From The Edge.mp3';
			'cure for the itch.mp3';
			'Darius Danesh - Colourblind.mp3';
			'daughtry_-_no_surprise.mp3';
			'Bryan Adams - Here I Am .mp3';
			'Eiffel 65 - Blue (da Ba Dee)_1.mp3';
			'Eminem - When I''m Gone.mp3';
			'Enrique Iglesias - Rhythm Divine.mp3';
			'Enya - Only Time (Sweet November).mp3';
			'Enya_-_I_Want_Tomorrow.mp3';
			'Evanescence-Bring Me to Life.mp3';
			'Evanescence-Everybody''s Fool.mp3';
			'Evanescence-Going Under.mp3';
			'Evanescence-My Immortal.mp3';
			'Evanescence-My Last Breath.mp3';
			'Evanescence-Taking Over Me.mp3';
			'Evanescence-Tourniquet.mp3';
			'faint.mp3';
			'Fort Minor - Remember the Name.mp3';
			'Frankie Goes To hollywood-Relax.mp3';
			'Hello I Love You.mp3';
			'Here Without You.mp3';
			'Hero - Chad Kroeger Ft Josey Scott.mp3';
			'hinder- lips of an angel.mp3';
			'How To Save A Life.mp3';
			'I Feel You.mp3';
			'in the end.mp3';
			'in your eyes.mp3';
			'It''s my life.mp3';
			'jump_in_the_line.mp3';
			'Katy Perry - Starstruck.mp3';
			'kayne west - stronger.mp3';
			'Kryptonite.mp3';
			'KR_forever.mp3';
			'kylie_minogue_higher.mp3';
			'Late Goodbye (Theme from Max Payne 2).mp3';
			'Lift.mp3';
			'Linkin Park & Britney Spears - Faint & Toxic.mp3';
			'Lollipop.mp3';
			'Lost Prophets - Hello Again.mp3';
			'Lost Prophets - Sway.mp3';
			'Lost prophets - Wake Up (Make A Move).mp3';
			'lyla.mp3';
			'Massive Attack - Teardrop.mp3';
			'metalica - metallica - nothing else matters.mp3';
			'metallica - i disappear.mp3';
			'Michael Jackson - Smooth Criminal .mp3';
			'Michael Jackson - They don''t care about us.mp3';
			'Michael jakson - Micheal jackson - Billie jean.mp3';
			'micheal jackson - michael jackson - beat it.mp3';
			'Micky.mp3';
			'Milk_And_Honey_-_Didi.mp3';
			'mOBSCENE.mp3';
			'moby - lift me up.mp3';
			'My Sacrifice - Creed.mp3';
			'numb.mp3';
			'Oasis - Stop Crying Your Heart Out.mp3';
			'Oasis - Wonderwall.mp3';
			'paint_the_sky_with_stars__the_best_of_enya--enya--anywhere_i.mp3';
			'paint_the_sky_with_stars__the_best_of_enya--enya--caribbean_.mp3';
			'paint_the_sky_with_stars__the_best_of_enya--enya--china_rose.mp3';
			'paint_the_sky_with_stars__the_best_of_enya--enya--only_if.mp3';
			'paint_the_sky_with_stars__the_best_of_enya--enya--on_my_way_.mp3';
			'paint_the_sky_with_stars__the_best_of_enya--enya--orinoco_fl.mp3';
			'Paradise City.mp3';
			'place for my head.mp3';
			'Poets of the Fall - Carnival Of Rust.mp3';
			'points of authority.mp3';
			'push tempo.mp3';
			'pushing me away.mp3';
			'Rock Is Dead.mp3';
			'runaway.mp3';
			'Santana feat. Chad Kroeger - Into the night.mp3';
			'sirther.mp3';
			'smash mouth - all-star.mp3';
			'Sweet Child O` Mine.mp3';
			'Tainted Love.mp3';
			'Take That - Patience.mp3';
			'The All American Rejects - Dirty Little Secret.mp3';
			'The Bannana Boat Song - Harry Belafonte.mp3';
			'The Beautiful People.mp3';
			'The Corrs - Unplugged - Toss The Feathers.mp3';
			'the felling -  Sewn.mp3';
			'The Fight Song.mp3';
			'The Roots and BT - Tao Of The Machine (Scott Humphrey''s Remix).mp3';
			'Things I''ll Never Say.mp3';
			'To Be Loved.mp3';
			'top of the world.mp3';
			'Turn the Page.mp3';
			'U2 - City of Blinding Lights.mp3';
			'U2 - Elevation.mp3';
			'U2 - Vertigo.mp3';
			'War, What is it Good For 1.mp3';
			'what goes around.mp3';
			'What I''ve Done.mp3';
			'When I''m Gone.mp3';
			'Whiskey Lullaby Feat. Alison Krauss.mp3';
			'x-good_charlotte_-_i_just_wanna_live.mp3';
            '01 - Mandy Moore - Cry.mp3'};
        
GmmDimension = 1;           % let GmmDimension be the number of dimensions used by the Gaussian model

% read mp3 files

for i = 1:N

	Songs(i)
    
    % read the sample data into a temporary vairable 'y'
    [y, fs] = mp3read(char(Songs(i)));
    
    %get the attributes
    
    ceps = mfcc(y,fs,100);   % get coefficients
                                % convert to gaussian mizxed model
    gm = gmdistribution.fit(ceps',GmmDimension);   

    % now gm.mu will contain one value per dimension. this is the vector we will
    % use your comparison

    Attributes(i,:) = gm.mu;

end

save('get_attributes.mat');
