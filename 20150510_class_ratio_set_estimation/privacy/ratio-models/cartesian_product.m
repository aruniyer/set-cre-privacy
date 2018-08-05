function result = cartesian_product(sets)

    if(size(sets, 1) == 1)
        result = sets{1}(:);
        return;
    end
    sets = cellfun(@unique, sets, 'UniformOutput', false);
    c = cell(1, numel(sets));
    [c{:}] = ndgrid(sets{:});
    result = cell2mat(cellfun(@(v)v(:), c, 'UniformOutput', false));
    
end