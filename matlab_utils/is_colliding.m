function colliding = is_colliding(obj1, obj2)
%IS_COLLIDING_AX1 Checks if obj1 and obj2 are colliding with separating axis theorem
    % obj1 and obj2 must be convex, and described by adjacent vertexes (ANTI-CLOCKWISE)
    % (meaning obj1(1) is assumed to be connected only to obj1(2) and
    % obj1(end)

    % Maybe AABB followed by SAT would be faster - if performance is an
    % issue - due to most objects not colliding with each other
    if ~ is_colliding_ax1(obj1, obj2)
        colliding = false;
        return
    end
    if ~ is_colliding_ax1(obj2, obj1)
        colliding = false;
        return
    end
    colliding = true;
end

function colliding = is_colliding_ax1(obj1, obj2)

    for i = 1:length(obj1)
        j = mod(i,length(obj1)) + 1;
        x1 = obj1(i, 1);
        y1 = obj1(i, 2);
        x2 = obj1(j, 1);
        y2 = obj1(j, 2);
        dx = x2 - x1;
        dy = y2 - y1;

        % find axis to project onto (perpendicular to edge)
        projection_axis = [dy, -dx]/norm([dy,dx]);

        % find projection of obj1 onto axis
        obj1_projection = dot(obj1, repmat(projection_axis, size(obj1, 1), 1), 2);

        % find projection of obj2 onto axis
        obj2_projection = dot(obj2, repmat(projection_axis, size(obj2, 1), 1), 2);

        % find "shadow" of projections
        p1m = min(obj1_projection);
        p1M = max(obj1_projection);
        p2m = min(obj2_projection);
        p2M = max(obj2_projection);

        % if projections don't overlap, a separating axis was found, 
        % so objects are not colliding
        if p1M < p2m || p1m > p2M
            colliding = false;
            return
        end
    end
    % if we get here, then for all edges, there was no separating axis
    colliding = true;

end

