#ifndef MPI_TAGS_H
#define MPI_TAGS_H

namespace octorl {
      enum tags {
        model_tag,
        batch_tag,
        keep_running_tag,
        gradient_tag,
        gradient_sync_tag
    };
}


#endif