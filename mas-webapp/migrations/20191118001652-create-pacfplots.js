'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.createTable('PACFPlots', {
      Id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      ModelId: {
        type: Sequelize.INTEGER,
        references: {
          model: 'ModelRunDetail',
          key: 'Id'
        }
      },
      Name : Sequelize.STRING,
      PLOT: Sequelize.BLOB
    });
  },

  down: (queryInterface, Sequelize) => {
    return queryInterface.dropTable('PACFPlots');
  }
};
