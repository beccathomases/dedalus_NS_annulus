function make_dedalus_movie()
%MAKE_DEDALUS_MOVIE  Make an MP4 from Dedalus v3 snapshots (HDF5).
%
% Assumes files like snapshots/snapshots_s1.h5, snapshots_s2.h5, ...

basedir = 'Re40_mpitest/snapshots';
task    = 'vorticity';    % 'vorticity' | 'divu' | 'p' | 'u'
ucomp   = 1;              % if task='u': 1=u_phi, 2=u_r
outname = sprintf('%s_movie.mp4', task);

files = dir(fullfile(basedir, 'snapshots_s*.h5'));
assert(~isempty(files), 'No snapshots_s*.h5 found in %s', basedir);
[~,ix] = sort({files.name});
files = files(ix);

vw = VideoWriter(outname, 'MPEG-4');
vw.FrameRate = 10;
open(vw);

fig = figure('Color','w');
ax  = axes(fig);

try
    for f = 1:numel(files)
        fname = fullfile(files(f).folder, files(f).name);

        % --- find sim_time dataset and read it ---
        t_path = find_dataset_path(fname, '/scales', 'sim_time');
        t      = squeeze(h5read(fname, t_path));
        t      = t(:);
        nW     = numel(t);

        % --- read task data ---
        if strcmp(task,'u')
            U = h5read(fname, '/tasks/u');
            Araw = extract_u_component(U, ucomp);
        else
            Araw = h5read(fname, ['/tasks/' task]);
        end

        % --- reshape to [Nphi x Nr x nW] ---
        A = reshape_to_phir_time(Araw, nW);

        % --- read phi/r coordinate vectors robustly ---
        % Prefer datasets under /scales that contain 'phi' or 'r' in their path/name.
        phi = read_coord_vector(fname, 'phi', size(A,1), size(A,2));
        r   = read_coord_vector(fname, 'r',   size(A,1), size(A,2));

        % Build annulus mesh
        [PHI, R] = ndgrid(phi(:), r(:));
        X = R .* cos(PHI);
        Y = R .* sin(PHI);

        % --- frame loop ---
        for k = 1:nW
            frame = A(:,:,k);

            cla(ax);
            surf(ax, X, Y, 0*X, frame, 'EdgeColor','none');
            view(ax, 2);
            axis(ax, 'equal', 'tight');
            colorbar(ax);
            title(ax, sprintf('%s   t = %.3f', task, t(k)));

            drawnow;
            writeVideo(vw, getframe(fig));
        end
    end
catch ME
    % Ensure video file closes cleanly even on error
    close(vw);
    rethrow(ME);
end

close(vw);
fprintf('Wrote %s\n', outname);

end

% ---------------- helpers ----------------

function paths = list_all_datasets(fname, groupPath)
% Recursively list full dataset paths under groupPath.
info = h5info(fname, groupPath);
paths = {};

% datasets directly in this group
for i = 1:numel(info.Datasets)
    ds = info.Datasets(i).Name;
    if strcmp(groupPath, '/')
        paths{end+1} = ['/' ds]; %#ok<AGROW>
    else
        paths{end+1} = [groupPath '/' ds]; %#ok<AGROW>
    end
end

% recurse into subgroups
for i = 1:numel(info.Groups)
    gname = info.Groups(i).Name;
    sub = list_all_datasets(fname, gname);
    paths = [paths, sub]; %#ok<AGROW>
end
end

function p = find_dataset_path(fname, baseGroup, mustContain)
% Find a dataset path under baseGroup that contains mustContain.
paths = list_all_datasets(fname, baseGroup);
hit = paths(~cellfun(@isempty, strfind(paths, mustContain)));
assert(~isempty(hit), 'Could not find dataset containing "%s" under %s', mustContain, baseGroup);
% Prefer exact-ish matches if multiple
p = hit{1};
for i = 1:numel(hit)
    if ~isempty(strfind(hit{i}, ['/' mustContain]))
        p = hit{i}; return;
    end
end
end

function v = read_coord_vector(fname, coordname, N1, N2)
% Try to find a 1D dataset under /scales related to coordname (phi or r).
paths = list_all_datasets(fname, '/scales');

% candidates containing coordname
cand = paths(~cellfun(@isempty, strfind(paths, coordname)));
% keep only 1D datasets
good = {};
for i = 1:numel(cand)
    info = h5info(fname, cand{i});
    sz = info.Dataspace.Size;
    if isscalar(sz) || (numel(sz)==2 && any(sz==1))
        good{end+1} = cand{i}; %#ok<AGROW>
    end
end
assert(~isempty(good), 'Could not find a 1D %s coordinate dataset under /scales', coordname);

% Prefer a vector whose length matches one of the spatial dims
targetLens = unique([N1, N2]);
best = good{1};
bestScore = -inf;

for i = 1:numel(good)
    vtmp = squeeze(h5read(fname, good{i}));
    vtmp = vtmp(:);
    score = 0;

    if any(numel(vtmp) == targetLens), score = score + 10; end
    if ~isempty(strfind(good{i}, '/1.0')), score = score + 2; end
    if ~isempty(strfind(good{i}, ['/scales/' coordname '/'])), score = score + 2; end

    if score > bestScore
        bestScore = score;
        best = good{i};
    end
end

v = squeeze(h5read(fname, best));
v = v(:);
end

function A = reshape_to_phir_time(Araw, nW)
A = squeeze(Araw);
sz = size(A);

% find time dimension (matches nW)
tDim = find(sz == nW, 1, 'first');
assert(~isempty(tDim), 'Could not find time dimension (nW=%d) in size %s', nW, mat2str(sz));

% move time to 3rd dim, and try to end up with [Nphi x Nr x nW]
% First, permute so time is last:
perm = 1:ndims(A);
perm([tDim, ndims(A)]) = perm([ndims(A), tDim]);
A = permute(A, perm);
A = squeeze(A);

% If time is now last, ensure 3D
if ndims(A) ~= 3
    % sometimes we get singleton dims; squeeze again
    A = squeeze(A);
end
assert(ndims(A)==3, 'Expected 3D array after reshaping; got size %s', mat2str(size(A)));

% Ensure time is 3rd
if size(A,3) ~= nW
    % maybe time ended up first; rotate dims
    if size(A,1)==nW
        A = permute(A, [2 3 1]);
    elseif size(A,2)==nW
        A = permute(A, [1 3 2]);
    end
end
assert(size(A,3)==nW, 'Could not place time in 3rd dim; final size %s', mat2str(size(A)));
end

function A = extract_u_component(U, ucomp)
U = squeeze(U);
sz = size(U);

% find component dim (size 2)
cDim = find(sz == 2, 1, 'last');
assert(~isempty(cDim), 'Could not find component dimension (size 2) in u of size %s', mat2str(sz));

subs = repmat({':'}, 1, ndims(U));
subs{cDim} = ucomp;
A = U(subs{:});
end